"""Thin multi-provider LLM wrapper with usage estimates for budget/cost guards."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from research_agent.token_utils import estimate_tokens


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    def __init__(self) -> None:
        self.provider = self._resolve_provider()
        self.model = (os.getenv("LLM_MODEL") or self._default_model()).strip()
        self._price_in = _env_float("LLM_PRICE_INPUT_PER_1K", 0.0)
        self._price_out = _env_float("LLM_PRICE_OUTPUT_PER_1K", 0.0)

    @staticmethod
    def _resolve_provider() -> str:
        explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if explicit:
            return explicit
        # Project default: Mistral-first.
        if os.getenv("MISTRAL_API_KEY"):
            return "mistral"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "mistral"

    def _default_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
        if self.provider == "mistral":
            return "mistral-small-latest"
        return "gpt-4o-mini"

    def estimate_cost_usd(self, result: LLMResult) -> float:
        if self._price_in <= 0 and self._price_out <= 0:
            return 0.0
        return (result.prompt_tokens / 1000.0) * self._price_in + (
            result.completion_tokens / 1000.0
        ) * self._price_out

    def complete(
        self,
        system: str,
        user: str,
        max_output_tokens: int,
        *,
        json_mode: bool = False,
    ) -> LLMResult:
        if os.getenv("RESEARCH_USE_MOCK_LLM", "").strip().lower() in {"1", "true", "yes", "on"}:
            return self._mock_complete(system, user, max_output_tokens, json_mode=json_mode)
        if self.provider == "openai":
            return self._openai(system, user, max_output_tokens, json_mode=json_mode)
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_output_tokens)
        if self.provider == "mistral":
            return self._mistral(system, user, max_output_tokens, json_mode=json_mode)
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _mock_complete(self, system: str, user: str, max_output_tokens: int, *, json_mode: bool) -> LLMResult:
        """
        Deterministic offline behavior for demos/CI.

        Goal: still demonstrate an end-to-end *answering* pipeline without any external LLM API.
        We do this by synthesizing a simple answer from the retrieved-memory blob that the
        agent passes into the final synthesis prompt.
        """

        def _extract_memory_sections(blob: str) -> list[tuple[str, str]]:
            # Matches the memory format emitted by MemoryStore.select_context().
            # Example heading: "### search_hit:2.1.1 (step 2, observation)"
            pat = re.compile(r"^###\s+([^\n(]+)(?:\s*\(.*\))?\n", re.MULTILINE)
            matches = list(pat.finditer(blob))
            out: list[tuple[str, str]] = []
            for i, m in enumerate(matches):
                src = m.group(1).strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
                body = blob[start:end].strip()
                if body:
                    out.append((src, body))
            return out

        def _synthesize_from_memory(prompt: str) -> str:
            # Pull the retrieved memory block (everything after "## Retrieved memory" if present).
            mem = prompt
            if "## Retrieved memory" in prompt:
                mem = prompt.split("## Retrieved memory", 1)[1]
            # Drop trailing instructions that appear after the retrieved memory blob.
            # (Prevents evidence bullets from leaking prompt directives.)
            for marker in (
                "\nProduce the final answer now.",
                "\nProduce the final answer now",
                "\nProduce the final answer",
            ):
                if marker in mem:
                    mem = mem.split(marker, 1)[0]
            sections = _extract_memory_sections(mem)
            cites = [src for src, _ in sections[:4]]
            joined = "\n".join(text for _, text in sections).lower()

            # Heuristics aimed at the canned demo prompt (RAG definition + failure mode).
            rag_def = (
                "Retrieval-augmented generation (RAG) is a method where a model retrieves relevant external "
                "documents first, then uses them as context to generate a grounded answer."
            )
            failure = "A common naive RAG failure mode is retrieving irrelevant/noisy chunks (or missing key context)."
            mitigation = "A common mitigation is improving chunking and retrieval (better chunking, reranking, filtering)."
            if "chunk" not in joined and "retriev" not in joined:
                mitigation = "A common mitigation is improving retrieval quality (better indexing, reranking, and filtering)."

            evidence_lines = []
            for src, body in sections[:6]:
                # keep short and readable
                snippet = re.sub(r"\s+", " ", body)[:220].strip()
                evidence_lines.append(f"- {snippet} ({src})")

            return "\n".join(
                [
                    "**Executive answer**",
                    f"- (1) {rag_def}",
                    f"- (2) {failure} {mitigation}",
                    "",
                    "**Evidence (from retrieved memory)**",
                    *(evidence_lines or [f"- (no evidence captured; cites: {', '.join(cites)})"]),
                ]
            ).strip()

        if json_mode or "Return strict JSON" in system:
            q = user.split("User question:", 1)[-1].strip() or user.strip()
            obj = {
                "sub_questions": [
                    {"id": 1, "question": q, "focus": "answer the user question directly"},
                    {"id": 2, "question": f"Failure modes / mitigations for: {q}", "focus": "find common pitfalls and fixes"},
                ]
            }
            text = json.dumps(obj)
        elif "Answer using ONLY the provided retrieved memory" in system or "## Retrieved memory" in user:
            text = _synthesize_from_memory(user)
        elif "compress" in system.lower() and "research notes" in system.lower():
            # Cheap summarizer: keep first ~8 bullets/lines.
            lines = [ln.strip() for ln in user.splitlines() if ln.strip()]
            text = "\n".join(lines[:16])[:4000]
        else:
            text = (
                "SUBQUESTION_DONE\n"
                "- Offline mode: tools ran and evidence was stored.\n"
                "- Final answer will be synthesized from retrieved memory.\n"
            )
        pt = estimate_tokens(system + user, self.model)
        ct = estimate_tokens(text, self.model)
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct)

    def _openai(self, system: str, user: str, max_tokens: int, *, json_mode: bool) -> LLMResult:
        from openai import OpenAI

        client = OpenAI()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        pt = int(usage.prompt_tokens) if usage else estimate_tokens(system + user, self.model)
        ct = int(usage.completion_tokens) if usage else estimate_tokens(text, self.model)
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct)

    def _anthropic(self, system: str, user: str, max_tokens: int) -> LLMResult:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for block in msg.content:
            if getattr(block, "text", None):
                parts.append(block.text)
        text = "".join(parts).strip()
        pt = int(getattr(msg.usage, "input_tokens", 0) or 0)
        ct = int(getattr(msg.usage, "output_tokens", 0) or 0)
        if pt == 0:
            pt = estimate_tokens(system + user, self.model)
        if ct == 0:
            ct = estimate_tokens(text, self.model)
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct)

    def _mistral(self, system: str, user: str, max_tokens: int, *, json_mode: bool) -> LLMResult:
        from mistralai.client import Mistral
        from mistralai.client import models as mm

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        msgs = [
            mm.SystemMessage(content=system),
            mm.UserMessage(content=user),
        ]
        response_format = None
        if json_mode:
            # Mistral supports JSON mode via response_format where available.
            # If the model ignores it, our caller still does tolerant JSON parsing.
            try:
                response_format = {"type": "json_object"}
            except Exception:
                response_format = None

        resp = client.chat.complete(
            model=self.model,
            messages=msgs,
            temperature=0.2,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        text = ""
        try:
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            text = str(getattr(resp, "text", "")).strip()

        pt = 0
        ct = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
        if pt == 0:
            pt = estimate_tokens(system + user, self.model)
        if ct == 0:
            ct = estimate_tokens(text, self.model)
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct)

    def complete_json_object(self, system: str, user: str, max_output_tokens: int) -> tuple[dict[str, Any], LLMResult]:
        if self.provider == "openai":
            res = self._openai(system, user, max_output_tokens, json_mode=True)
        else:
            sys2 = system + "\nRespond with a single JSON object only. No markdown fences."
            res = self.complete(sys2, user, max_output_tokens, json_mode=False)
        data = self._loads_json_object(res.text)
        return data, res

    @staticmethod
    def _loads_json_object(text: str) -> dict[str, Any]:
        raw = text.strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        t = raw
        if "```" in t:
            parts = t.split("```", 2)
            if len(parts) >= 2:
                inner = parts[1]
                if inner.lower().lstrip().startswith("json"):
                    inner = inner.lstrip()[4:].lstrip()
                t = inner
        obj = json.loads(t)
        if not isinstance(obj, dict):
            raise TypeError("Expected JSON object")
        return obj
