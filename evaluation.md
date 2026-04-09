# G3 evaluation: memory architecture and trade-offs

This note documents the **deliberate constraints**, what we optimized for, and what we gave up.

## Stated constraints (self-defined)

- **`max_context_tokens_per_llm_call` (default: 2000 token-estimate units):** caps how much *stored research evidence* can be injected into the final synthesis prompt after retrieval scoring/packing.
- **`max_session_memory_tokens` (default: 8000):** caps the raw episodic store; when exceeded, the agent runs a **summarization cascade** that merges older observations into a compact summary chunk.
- **Optional `max_cost_usd_per_session`:** if set along with `LLM_PRICE_INPUT_PER_1K` / `LLM_PRICE_OUTPUT_PER_1K`, the session stops once estimated spend crosses the ceiling.

Token estimates prefer **tiktoken** when available; otherwise a **~4 characters ~ 1 token** heuristic is used (documented fallback).

## Workflow orchestration (n8n / Dify)

The **research/memory policy** still executes inside Python (so constraints stay centralized), while **n8n** handles ingress/scheduling/retries:

- n8n **Webhook** -> **HTTP Request** -> `POST /research` on the FastAPI app.
- Dify can mirror the same pattern with an HTTP tool node.

This matches the brief's "query routing + memory management" intent: the workflow **routes** requests and can persist responses downstream (Sheets, DB, email) without re-implementing memory logic in the visual layer.

In practical terms, this is aimed at use cases like:

- **Analyst support:** take a research prompt from a form or webhook, run a bounded research session, then forward the structured result to a dashboard, sheet, or approval queue.
- **Operations / client-facing research requests:** keep each request within a predictable budget while still returning an answer plus traceable evidence metadata.

## Memory strategy (what we implemented)

We use a **hybrid episodic buffer + summarization cascade + budgeted retrieval (BM25 ranking)**:

1. **Episodic buffer:** each tool output becomes an `observation` chunk tagged with `source`, `step`, and token estimate.
2. **Compaction:** if total stored tokens exceed `max_session_memory_tokens`, the oldest half of observations are merged via an LLM summarization call into a single `summary` chunk (repeat until under budget or safety stop).
3. **Final retrieval:** for synthesis, chunks are **BM25-ranked** then **greedy-packed** into a `max_context_tokens_per_llm_call` budget (with a mild recency tie-break).

## Why not "classic vector RAG" here?

**Vector retrieval** (embeddings + cosine similarity) is strong when you have a stable corpus and repeated queries. For this assignment prototype, web snippets are **tiny, noisy, and frequently redundant**. Embeddings add:

- Extra dependencies and operational complexity (embedding provider costs, latency, dimensionality choices)
- Another failure mode (embedding endpoint outages, rate limits)

The chosen **local BM25** retriever is easier to reason about under a hard context cap: it's deterministic-ish, cheap, and fails "openly" (it may miss paraphrases - see trade-offs).

## Trade-offs (what you gain / lose)

| Choice | Upside | Downside |
| --- | --- | --- |
| Budgeted packing | Guarantees we *attempt* to respect a hard "injected context" budget | Packing is greedy; not globally optimal |
| **BM25 scoring** | Still local + cheap, better ranking than raw overlap | Weaker on synonyms/paraphrase vs embeddings |
| Summarization cascade | Keeps long sessions bounded | Summary can drop rare details; risk of hallucinated compression if the model overreaches |
| Web tools (DDG + fetch) | Demonstrates realistic research loops | Live search can be flaky by region/network; mock mode exists for reproducibility |
| **Mistral tool calling** | Satisfies "tool use" requirement with a clear tool loop | Model can request many calls; we cap rounds and enforce tool budgets |
| Offline mock mode | Reproducible demos without API keys | Not equivalent to a real model's reasoning quality |
| **n8n + FastAPI** | Clear operational story for reviewers; easy scheduling | Requires running two services locally for demos |

## What "success" looks like in this repo

- The agent **decomposes** a multi-part question into sub-questions.
- Each sub-question triggers a **research cycle** (logs on stderr): search -> optional fetch -> memory append -> optional compaction.
- Final answering uses **only** budget-selected chunks, and the JSON output includes **which chunk IDs** were selected plus token estimates.

## Business impact (why a client would care)

In enterprise research workflows, the expensive failure mode is usually **unbounded context**: long tool transcripts get pasted into prompts, costs spike, and quality drops due to attention dilution. A **declared budget + explicit retrieval + compaction** turns research into an auditable pipeline: you can report what evidence was used and why it fit within client constraints.

That matters in settings where teams care about both **answer quality** and **operational discipline**. A client is more likely to trust a system that can say "these were the chunks used, this was the rough token footprint, and this session stayed within the declared limit" than one that simply returns a long answer with no cost or evidence story.

## If this were extended (next improvements)

- Replace lexical scoring with **cheap local embeddings** (or hybrid BM25 + embeddings) once the evidence corpus stabilizes.
- Add **citations resolution** (fetch canonical URLs, dedupe near-duplicates) before memory insert.
- Add **n8n/Dify** orchestration around the same stages for schedules, retries, and human approvals - without changing the core memory semantics.
