"""Orchestrates decomposition → tool use → constrained retrieval → synthesis."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from research_agent.config import ResearchConstraints
from research_agent.llm import LLMClient, LLMResult
from research_agent.memory import MemoryStore
from research_agent.mistral_tooling import run_mistral_subquestion_tools
from research_agent.tools import fetch_url_text, search_web
from research_agent.token_utils import estimate_tokens


class SubQuestion(BaseModel):
    id: int
    question: str
    focus: str = Field(description="What fact gap this sub-question closes")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        raise ValueError(f"Invalid id: {v!r}")


class Decomposition(BaseModel):
    sub_questions: list[SubQuestion]


@dataclass
class SessionLedger:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    events: list[str] = field(default_factory=list)

    def add(self, llm: LLMClient, res: LLMResult, note: str) -> None:
        self.total_prompt_tokens += res.prompt_tokens
        self.total_completion_tokens += res.completion_tokens
        self.total_cost_usd += llm.estimate_cost_usd(res)
        self.events.append(
            json.dumps(
                {
                    "llm_call": note,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "session_cost_usd": round(self.total_cost_usd, 6),
                }
            )
        )


def _log(line: str) -> None:
    print(line, file=sys.stderr)


def _parse_decomposition(payload: dict[str, Any]) -> Decomposition:
    return Decomposition.model_validate(payload)


def run_research(
    question: str,
    constraints: ResearchConstraints | None = None,
    *,
    use_mock_tools: bool = False,
) -> dict[str, Any]:
    prev_mock = os.environ.get("RESEARCH_USE_MOCK_TOOLS")
    if use_mock_tools:
        os.environ["RESEARCH_USE_MOCK_TOOLS"] = "1"
        # Also mock LLM calls so the demo is reproducible even without Gemini quota.
        os.environ["RESEARCH_USE_MOCK_LLM"] = "1"
    try:
        return _run_research_core(question, constraints)
    finally:
        if use_mock_tools:
            if prev_mock is None:
                os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)
            else:
                os.environ["RESEARCH_USE_MOCK_TOOLS"] = prev_mock
            os.environ.pop("RESEARCH_USE_MOCK_LLM", None)


def _run_research_core(question: str, constraints: ResearchConstraints | None = None) -> dict[str, Any]:
    c = constraints or ResearchConstraints()
    llm = LLMClient()
    ledger = SessionLedger()
    memory = MemoryStore(c, model_hint=llm.model)
    tooling_mode = (
        "mistral_tool_calling" if llm.provider == "mistral" else "imperative_direct_tools"
    )
    step_traces: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def _cost_guard() -> str | None:
        if c.max_cost_usd_per_session is None:
            return None
        if ledger.total_cost_usd > c.max_cost_usd_per_session:
            return "Session stopped: exceeded max_cost_usd_per_session."
        return None

    # 1) Decompose
    decompose_system = (
        "You plan web research. Break the user's question into non-overlapping sub-questions "
        "that can be answered from public web evidence. Prefer 2-4 sub-questions unless the prompt "
        "is truly atomic. Return strict JSON: "
        '{"sub_questions":[{"id":1,"question":"...","focus":"..."}, ...]}'
    )
    decompose_user = f"User question:\n{question}\n"
    try:
        raw_plan, dec_res = llm.complete_json_object(
            decompose_system, decompose_user, max_output_tokens=min(700, c.max_llm_output_tokens)
        )
        ledger.add(llm, dec_res, note="decompose")
        plan = _parse_decomposition(raw_plan)
    except Exception as exc:
        _log(f"[warn] decomposition_fallback reason={exc!r}")
        plan = Decomposition(
            sub_questions=[
                SubQuestion(id=1, question=question, focus="answer the user question directly")
            ]
        )
    plan.sub_questions = plan.sub_questions[: c.max_sub_questions]

    _log(
        json.dumps(
            {
                "event": "plan_ready",
                "sub_questions": [sq.model_dump() for sq in plan.sub_questions],
            }
        )
    )

    # 2) Iterative gather: each sub-question is one "research cycle"
    for idx, sq in enumerate(plan.sub_questions, start=1):
        reason = _cost_guard()
        if reason:
            _log(json.dumps({"event": "stopped", "reason": reason}))
            break

        _log(json.dumps({"event": "research_cycle_start", "step": idx, "sub_question": sq.question}))
        before_ids = {c.chunk_id for c in memory.chunks}
        step_t0 = time.perf_counter()
        trace: dict[str, Any] = {
            "step": idx,
            "sub_question": sq.question,
            "tooling_mode": tooling_mode,
            "tool_calls": {},
            "urls": {"searched": [], "fetched": []},
            "new_chunk_ids": [],
            "memory_tokens_after": None,
            "elapsed_ms": None,
        }

        if llm.provider == "mistral":
            state, _text, _res = run_mistral_subquestion_tools(
                model=llm.model,
                memory=memory,
                step=idx,
                sub_question=sq.question,
                constraints=c,
                cost_abort_fn=lambda: _cost_guard() is not None,
                ledger=ledger,
                llm=llm,
                log_fn=_log,
            )
            trace["tool_calls"] = state.get("tool_calls", {})
            trace["urls"] = state.get("urls", trace["urls"])
        else:
            hits = search_web(sq.question, max_results=c.max_search_results_per_subq)
            if not hits:
                memory.add_observation(
                    text=f"No search results returned for: {sq.question}",
                    source=f"search:{sq.id}",
                    step=idx,
                    meta={"sub_question": sq.question},
                )
            for j, h in enumerate(hits, start=1):
                snippet = f"{h.title}\nURL: {h.href}\nSnippet: {h.body}"
                memory.add_observation(
                    text=snippet,
                    source=f"search_hit:{idx}.1.{j}",
                    step=idx,
                    meta={"href": h.href, "title": h.title, "via": "imperative_tools"},
                )
                if h.href:
                    trace["urls"]["searched"].append(h.href)

            if c.max_url_fetches_per_subq > 0 and hits:
                url = next((h.href for h in hits if h.href.startswith("http")), "")
                if url:
                    page = fetch_url_text(url)
                    if page:
                        memory.add_observation(
                            text=f"Fetched page excerpt from {url}\n{page}",
                            source=f"fetch:{idx}",
                            step=idx,
                            meta={"url": url, "via": "imperative_tools"},
                        )
                        trace["urls"]["fetched"].append(url)
            trace["tool_calls"] = {
                "web_search": 1,
                "fetch_url": 1 if trace["urls"]["fetched"] else 0,
            }

        def _summarize(blob: str) -> str:
            if _cost_guard():
                return blob[:4000]
            sys_msg = "You compress research notes. Be faithful; do not invent facts."
            usr = blob
            out = llm.complete(
                sys_msg, usr, max_output_tokens=min(c.summary_target_tokens + 200, c.max_llm_output_tokens)
            )
            ledger.add(llm, out, note=f"summarize_step_{idx}")
            return out.text

        for line in memory.maybe_compact(_summarize):
            _log(line)

        after = memory.chunks
        trace["new_chunk_ids"] = [c.chunk_id for c in after if c.chunk_id not in before_ids]
        trace["memory_tokens_after"] = memory.total_tokens()
        trace["elapsed_ms"] = int((time.perf_counter() - step_t0) * 1000)
        step_traces.append(trace)

        _log(
            json.dumps(
                {
                    "event": "research_cycle_end",
                    "step": idx,
                    "memory_total_tokens": memory.total_tokens(),
                }
            )
        )

    # 3) Constrained retrieval for final synthesis
    reason = _cost_guard()
    if reason:
        memory_blob, picked_ids = memory.select_context(question, budget_tokens=c.max_context_tokens_per_llm_call)
        answer = (
            f"{reason}\n\nPartial context was retrieved under budget. "
            f"Selected chunk IDs: {', '.join(picked_ids) or '(none)'}"
        )
        return _package_result(
            question,
            plan,
            memory,
            memory_blob,
            answer,
            ledger,
            picked_ids,
            tooling_mode=tooling_mode,
            llm_provider=llm.provider,
            steps=step_traces,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    memory_blob, picked_ids = memory.select_context(question, budget_tokens=c.max_context_tokens_per_llm_call)

    syn_system = (
        "You are a careful research assistant. Answer using ONLY the provided retrieved memory. "
        "If memory is insufficient, say what is missing and answer partially. "
        "Cite sources by their source labels (e.g., search_hit:2.1). "
        "Structure: short executive answer, then bullet evidence with cites."
    )
    syn_user = (
        f"Original question:\n{question}\n\n"
        f"{memory_blob}\n\n"
        "Produce the final answer now."
    )

    # Ensure we don't accidentally explode context: trim user if needed
    over = estimate_tokens(syn_system + syn_user, llm.model) - int(c.max_context_tokens_per_llm_call * 1.5)
    if over > 0:
        syn_user = syn_user[: -min(len(syn_user), over * 4)]

    final = llm.complete(syn_system, syn_user, max_output_tokens=c.max_llm_output_tokens)
    ledger.add(llm, final, note="final_synthesis")

    return _package_result(
        question,
        plan,
        memory,
        memory_blob,
        final.text,
        ledger,
        picked_ids,
        tooling_mode=tooling_mode,
        llm_provider=llm.provider,
        steps=step_traces,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


def _package_result(
    question: str,
    plan: Decomposition,
    memory: MemoryStore,
    memory_blob: str,
    answer: str,
    ledger: SessionLedger,
    picked_ids: list[str],
    *,
    tooling_mode: str,
    llm_provider: str,
    steps: list[dict[str, Any]],
    elapsed_ms: int,
) -> dict[str, Any]:
    picked = memory.get_by_ids(picked_ids)
    picked_evidence = []
    for c in picked:
        url = c.meta.get("href") or c.meta.get("url")
        picked_evidence.append(
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "kind": c.kind,
                "step": c.step,
                "url": url,
                "title": c.meta.get("title"),
                "token_estimate": c.token_estimate,
            }
        )
    return {
        "question": question,
        "stack": {
            "llm_provider": llm_provider,
            "tool_use": tooling_mode,
            "memory": "episodic_buffer + summarization_cascade + budgeted_retrieval",
            "workflow_orchestration": "n8n (see /orchestration) calling FastAPI POST /research",
        },
        "execution": {
            "elapsed_ms": elapsed_ms,
            "steps": steps,
        },
        "sub_questions": [sq.model_dump() for sq in plan.sub_questions],
        "retrieval": {
            "selected_chunk_ids": picked_ids,
            "selected_context_token_estimate": estimate_tokens(memory_blob, "gpt-4o-mini"),
            "memory_store_token_total": memory.total_tokens(),
            "selected_evidence": picked_evidence,
        },
        "answer": answer,
        "session": {
            "prompt_tokens": ledger.total_prompt_tokens,
            "completion_tokens": ledger.total_completion_tokens,
            "estimated_cost_usd": round(ledger.total_cost_usd, 6),
            "events": ledger.events,
        },
    }