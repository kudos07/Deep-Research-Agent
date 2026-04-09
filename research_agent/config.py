"""Self-defined constraints for the research session (G3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchConstraints(BaseModel):
    """
    Budget model (token-like units):

    - We use tiktoken when available for ``text-embedding`` / OpenAI-style counting;
      otherwise ~4 chars ≈ 1 token (documented fallback).
    - ``max_context_tokens_per_llm_call`` caps *injected memory + tool snippets*
      per completion that feeds the final answer (retrieval discipline).
    - ``max_session_memory_tokens`` triggers a summarization cascade on the store
      so long runs stay bounded.
    - ``max_cost_usd_per_session`` is optional; if set, we stop after estimating
      spend from per-1k token rates (see .env.example).
    """

    max_context_tokens_per_llm_call: int = Field(
        default=2000,
        ge=500,
        description="Hard cap on retrieved memory (token estimate) injected into any single LLM call.",
    )
    max_session_memory_tokens: int = Field(
        default=8000,
        ge=2000,
        description="When the episodic store exceeds this, oldest chunks merge into a summary.",
    )
    summary_target_tokens: int = Field(
        default=450,
        ge=200,
        description="Target size for compressed rollup chunks.",
    )
    max_sub_questions: int = Field(default=5, ge=1, le=12)
    max_search_results_per_subq: int = Field(default=3, ge=1, le=8)
    max_url_fetches_per_subq: int = Field(default=1, ge=0, le=3)
    max_llm_output_tokens: int = Field(default=900, ge=200)
    max_cost_usd_per_session: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional USD ceiling; requires LLM_*_PRICE_PER_1K env vars.",
    )
