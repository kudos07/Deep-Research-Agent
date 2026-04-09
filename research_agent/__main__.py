"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from research_agent import __version__
from research_agent.agent import run_research
from research_agent.config import ResearchConstraints


def _has_llm_key() -> bool:
    return bool(os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="research_agent", description="G3 constrained-memory research agent")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a single research question")
    r.add_argument("question", help="Multi-part research question")
    r.add_argument("--max-context", type=int, default=None, help="Override max_context_tokens_per_llm_call")
    r.add_argument("--max-memory", type=int, default=None, help="Override max_session_memory_tokens")
    r.add_argument("--max-cost-usd", type=float, default=None, help="Set max_cost_usd_per_session (use 0 to disable)")
    r.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
    r.add_argument(
        "--mock-tools",
        action="store_true",
        help="Use deterministic mock search/fetch output (recommended for offline demos)",
    )

    d = sub.add_parser("demo", help="Run a canned multi-part question (shows multiple research cycles in logs)")
    d.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
    d.add_argument(
        "--mock-tools",
        action="store_true",
        help="Use deterministic mock search/fetch output (recommended for offline demos)",
    )

    args = p.parse_args(argv)

    if not _has_llm_key() and not getattr(args, "mock_tools", False):
        print(
            "Missing API key. For the intended G3 stack, set MISTRAL_API_KEY (or set LLM_PROVIDER + another provider key).",
            file=sys.stderr,
        )
        return 2

    if args.cmd == "demo":
        question = (
            "In two parts: (1) What is retrieval-augmented generation (RAG) in one sentence? "
            "(2) Name one common failure mode of naive RAG and a mitigation mentioned in public discussions."
        )
        constraints = ResearchConstraints(
            max_context_tokens_per_llm_call=2000,
            max_session_memory_tokens=8000,
            max_sub_questions=4,
        )
        out = run_research(question, constraints=constraints, use_mock_tools=args.mock_tools)
        pretty = bool(args.pretty)
    else:
        constraints = ResearchConstraints()
        if args.max_context is not None:
            constraints.max_context_tokens_per_llm_call = args.max_context
        if args.max_memory is not None:
            constraints.max_session_memory_tokens = args.max_memory
        if args.max_cost_usd is not None:
            constraints.max_cost_usd_per_session = None if args.max_cost_usd <= 0 else float(args.max_cost_usd)
        out = run_research(args.question, constraints=constraints, use_mock_tools=args.mock_tools)
        pretty = bool(args.pretty)

    if pretty:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
