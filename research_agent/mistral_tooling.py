"""Mistral tool-calling for web_search + fetch_url (local tool execution)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from mistralai.client import Mistral
from mistralai.client import models as mm

from research_agent.config import ResearchConstraints
from research_agent.llm import LLMClient, LLMResult
from research_agent.memory import MemoryStore
from research_agent.tools import fetch_url_text, search_web
from research_agent.token_utils import estimate_tokens


def run_mistral_subquestion_tools(
    *,
    model: str,
    memory: MemoryStore,
    step: int,
    sub_question: str,
    constraints: ResearchConstraints,
    max_rounds: int = 8,
    cost_abort_fn: Callable[[], bool] | None = None,
    ledger: Any | None = None,
    llm: LLMClient | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str, LLMResult]:
    """
    One research cycle using Mistral's tool-calling.

    We provide tool schemas to the model, execute requested tool calls locally,
    store results into episodic memory, then continue until no tool calls remain.
    """

    state: dict[str, Any] = {
        "tool_calls": {"web_search": 0, "fetch_url": 0},
        "urls": {"searched": [], "fetched": []},
        "errors": [],
    }

    def web_search(query: str, max_results: int = 3) -> dict:
        if cost_abort_fn and cost_abort_fn():
            return {"error": "session_cost_guard_triggered"}
        state["tool_calls"]["web_search"] += 1
        call_no = int(state["tool_calls"]["web_search"])
        mx = max(1, min(int(max_results), constraints.max_search_results_per_subq))
        hits = search_web(query.strip(), max_results=mx)
        rows: list[dict[str, str]] = []
        for j, h in enumerate(hits, start=1):
            snippet = f"{h.title}\nURL: {h.href}\nSnippet: {h.body}"
            memory.add_observation(
                text=snippet,
                source=f"search_hit:{step}.{call_no}.{j}",
                step=step,
                meta={
                    "href": h.href,
                    "title": h.title,
                    "via": "mistral_tool_call",
                    "query": query,
                    "search_call_no": call_no,
                    "rank": j,
                },
            )
            if h.href:
                state["urls"]["searched"].append(h.href)
            rows.append({"title": h.title, "url": h.href, "snippet": h.body})
        return {"results": rows}

    def fetch_url(url: str) -> dict:
        if cost_abort_fn and cost_abort_fn():
            return {"error": "session_cost_guard_triggered"}
        state["tool_calls"]["fetch_url"] += 1
        call_no = int(state["tool_calls"]["fetch_url"])
        if call_no > constraints.max_url_fetches_per_subq:
            state["errors"].append("fetch_url_budget_exceeded_for_subquestion")
            return {"error": "fetch_url_budget_exceeded_for_subquestion"}
        page = fetch_url_text(url.strip())
        if not page:
            memory.add_observation(
                text=f"fetch_url returned empty body for: {url}",
                source=f"tool:fetch_empty:{step}.{call_no}",
                step=step,
                meta={"url": url, "via": "mistral_tool_call", "fetch_call_no": call_no},
            )
            return {"extract": ""}
        memory.add_observation(
            text=f"Fetched page excerpt from {url}\n{page}",
            source=f"fetch:{step}.{call_no}",
            step=step,
            meta={"url": url, "via": "mistral_tool_call", "fetch_call_no": call_no},
        )
        state["urls"]["fetched"].append(url)
        return {"extract_chars": len(page), "url": url}

    # Offline mode: skip external API calls but still capture tool evidence.
    if os.getenv("RESEARCH_USE_MOCK_LLM", "").strip().lower() in {"1", "true", "yes", "on"}:
        state["errors"].append("mock_llm_enabled")
        web_search(sub_question, max_results=constraints.max_search_results_per_subq)
        if constraints.max_url_fetches_per_subq > 0:
            first = (state["urls"]["searched"] or ["https://example.org/mock"])[0]
            fetch_url(first)
        text = "SUBQUESTION_DONE\n- Tool evidence captured in memory (mock LLM).\n"
        res = LLMResult(text=text, prompt_tokens=1, completion_tokens=1)
        return state, text, res

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    tools = [
        mm.Tool(
            function=mm.Function(
                name="web_search",
                description="Search the web and return snippets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
            )
        ),
        mm.Tool(
            function=mm.Function(
                name="fetch_url",
                description="Fetch a URL and store an excerpt.",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            )
        ),
    ]

    system = (
        "You are a disciplined research assistant. Use tools to gather evidence, then stop.\n"
        "When finished, output:\n"
        "Line 1: SUBQUESTION_DONE\n"
        "Then 2-6 bullet points with supported claims and the URLs used.\n"
        "Do not invent URLs."
    )

    messages: list[Any] = [
        mm.SystemMessage(content=system),
        mm.UserMessage(content=f"Sub-question ({step}): {sub_question}"),
    ]

    if log_fn:
        log_fn(json.dumps({"event": "mistral_tool_session_start", "step": step}))

    rounds = 0
    final_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    while True:
        rounds += 1
        if rounds > max_rounds:
            state["errors"].append("tool_round_budget_exceeded")
            break

        resp = client.chat.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=700,
        )

        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            # Add assistant message containing the tool calls
            messages.append(mm.AssistantMessage(content=msg.content, tool_calls=tool_calls))

            for tc in tool_calls:
                fn = tc.function
                name = fn.name
                args_raw = fn.arguments
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except Exception:
                    args = {}
                if name == "web_search":
                    out = web_search(**args)
                elif name == "fetch_url":
                    out = fetch_url(**args)
                else:
                    out = {"error": f"unknown_function:{name}"}
                    state["errors"].append(out["error"])

                messages.append(
                    mm.ToolMessage(
                        content=json.dumps(out),
                        tool_call_id=getattr(tc, "id", None),
                        name=name,
                    )
                )
            continue

        content = msg.content
        if isinstance(content, list):
            # Some SDK versions return multi-part content.
            parts: list[str] = []
            for p in content:
                if p is None:
                    continue
                if isinstance(p, str):
                    parts.append(p)
                else:
                    parts.append(str(p))
            final_text = "\n".join(parts).strip()
        else:
            final_text = (content or "").strip()
        break

    if final_text == "":
        final_text = "SUBQUESTION_DONE\n- No final text (tool loop ended).\n"

    if prompt_tokens == 0 and completion_tokens == 0:
        blob = "\n".join(str(m) for m in messages)
        prompt_tokens = estimate_tokens(blob, model)
        completion_tokens = estimate_tokens(final_text, model)

    result = LLMResult(text=final_text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    if ledger is not None and llm is not None:
        ledger.add(llm, result, note=f"mistral_tools_step_{step}")

    if log_fn:
        log_fn(
            json.dumps(
                {"event": "mistral_tool_session_end", "step": step, "usage": {"prompt": prompt_tokens, "completion": completion_tokens}}
            )
        )

    return state, final_text, result

