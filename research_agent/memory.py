"""Episodic buffer + budget-aware retrieval + summarization cascade."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

from research_agent.config import ResearchConstraints
from research_agent.token_utils import estimate_tokens

ChunkKind = Literal["observation", "summary"]


_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "as",
        "at",
        "by",
        "it",
        "this",
        "that",
        "from",
        "not",
    }
)


def _terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in _STOP
    }


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP]


def _bm25_scores(query: str, docs: list[str], k1: float = 1.4, b: float = 0.75) -> list[float]:
    """
    Lightweight BM25 for short snippets.
    Deterministic, local, and improves over simple overlap for ranking.
    """
    q = _tokens(query)
    if not q or not docs:
        return [0.0 for _ in docs]

    doc_tokens = [_tokens(d) for d in docs]
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0

    # document frequency
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    N = len(docs)
    scores: list[float] = []
    for toks, dl in zip(doc_tokens, doc_lens, strict=True):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for term in q:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            n = df.get(term, 0)
            idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
            denom = f + k1 * (1.0 - b + b * (dl / (avgdl or 1.0)))
            s += idf * (f * (k1 + 1.0)) / (denom or 1.0)
        scores.append(s)
    return scores


@dataclass
class MemoryChunk:
    chunk_id: str
    kind: ChunkKind
    text: str
    source: str
    step: int
    token_estimate: int
    meta: dict = field(default_factory=dict)


class MemoryStore:
    def __init__(self, constraints: ResearchConstraints, model_hint: str) -> None:
        self._c = constraints
        self._model_hint = model_hint
        self._chunks: list[MemoryChunk] = []

    @property
    def chunks(self) -> list[MemoryChunk]:
        return list(self._chunks)

    def get_by_ids(self, chunk_ids: list[str]) -> list[MemoryChunk]:
        wanted = set(chunk_ids)
        return [c for c in self._chunks if c.chunk_id in wanted]

    def total_tokens(self) -> int:
        return sum(c.token_estimate for c in self._chunks)

    def add_observation(self, text: str, source: str, step: int, meta: dict | None = None) -> MemoryChunk:
        tid = str(uuid.uuid4())
        est = estimate_tokens(text, self._model_hint)
        ch = MemoryChunk(
            chunk_id=tid,
            kind="observation",
            text=text.strip(),
            source=source,
            step=step,
            token_estimate=est,
            meta=dict(meta or {}),
        )
        self._chunks.append(ch)
        return ch

    def maybe_compact(self, summarize_fn: Callable[[str], str]) -> list[str]:
        """
        If total store exceeds max_session_memory_tokens, compress the oldest
        half of *observation* chunks into one summary chunk (cascade).
        Returns human-readable log lines.
        """
        logs: list[str] = []
        rounds = 0
        while self.total_tokens() > self._c.max_session_memory_tokens and rounds < 6:
            rounds += 1
            obs_idx = [i for i, c in enumerate(self._chunks) if c.kind == "observation"]
            if len(obs_idx) < 2:
                break
            take = max(2, len(obs_idx) // 2)
            steal = obs_idx[:take]
            stolen = [self._chunks[i] for i in steal]
            for i in reversed(steal):
                del self._chunks[i]
            blob = "\n\n".join(
                f"[{c.source} | step {c.step}]\n{c.text}" for c in stolen
            )
            prompt = (
                "Compress the following research excerpts into dense bullet notes. "
                "Preserve numbers, names, dates, and causal claims. "
                f"Keep under ~{self._c.summary_target_tokens} tokens.\n\n"
                f"{blob}"
            )
            try:
                summary_text = summarize_fn(prompt)
            except Exception as exc:  # pragma: no cover - defensive
                summary_text = f"(summarization failed: {exc})"
            summary = MemoryChunk(
                chunk_id=str(uuid.uuid4()),
                kind="summary",
                text=summary_text.strip(),
                source="memory:summarization_cascade",
                step=stolen[0].step,
                token_estimate=estimate_tokens(summary_text, self._model_hint),
                meta={"merged_chunk_ids": [c.chunk_id for c in stolen]},
            )
            self._chunks.append(summary)
            logs.append(
                json.dumps(
                    {
                        "event": "memory_compaction",
                        "merged_observations": len(stolen),
                        "new_summary_tokens": summary.token_estimate,
                        "store_total_tokens": self.total_tokens(),
                    }
                )
            )
        return logs

    def select_context(self, query: str, budget_tokens: int) -> tuple[str, list[str]]:
        """
        Greedy packing by relevance (BM25) then recency within budget.
        Returns (joined context, list of selected chunk ids).
        """
        docs = [c.text for c in self._chunks]
        base_scores = _bm25_scores(query, docs)
        scored: list[tuple[float, int, MemoryChunk]] = []
        for c, base in zip(self._chunks, base_scores, strict=True):
            score = float(base) + (0.01 * c.step)  # small recency tie-break
            scored.append((score, c.step, c))
        scored.sort(key=lambda t: (-t[0], -t[1]))

        picked: list[MemoryChunk] = []
        used = 0
        header = estimate_tokens("## Retrieved memory\n", self._model_hint)
        used += header

        for _, __, ch in scored:
            line = f"### {ch.source} (step {ch.step}, {ch.kind})\n{ch.text}\n\n"
            need = estimate_tokens(line, self._model_hint)
            if used + need > budget_tokens:
                continue
            picked.append(ch)
            used += need

        if not picked and self._chunks:
            # Ensure we never return empty if memory exists: take newest tiny slice
            last = self._chunks[-1]
            text = f"### {last.source}\n{last.text[: budget_tokens * 4]}\n"
            return text, [last.chunk_id]

        blob = "## Retrieved memory\n" + "".join(
            f"### {c.source} (step {c.step}, {c.kind})\n{c.text}\n\n" for c in picked
        )
        return blob, [c.chunk_id for c in picked]
