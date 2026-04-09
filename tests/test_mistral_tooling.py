import os
import unittest

from research_agent.config import ResearchConstraints
from research_agent.llm import LLMClient
from research_agent.memory import MemoryStore
from research_agent.mistral_tooling import run_mistral_subquestion_tools


class DummyLedger:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def add(self, llm, res, note: str) -> None:
        self.calls.append(note)


class MistralToolingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_mock_llm = os.environ.get("RESEARCH_USE_MOCK_LLM")
        self.saved_mock_tools = os.environ.get("RESEARCH_USE_MOCK_TOOLS")
        os.environ["RESEARCH_USE_MOCK_LLM"] = "1"
        os.environ["RESEARCH_USE_MOCK_TOOLS"] = "1"

    def tearDown(self) -> None:
        if self.saved_mock_llm is None:
            os.environ.pop("RESEARCH_USE_MOCK_LLM", None)
        else:
            os.environ["RESEARCH_USE_MOCK_LLM"] = self.saved_mock_llm
        if self.saved_mock_tools is None:
            os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)
        else:
            os.environ["RESEARCH_USE_MOCK_TOOLS"] = self.saved_mock_tools

    def test_mock_mode_collects_search_and_fetch_evidence(self) -> None:
        constraints = ResearchConstraints(max_url_fetches_per_subq=1)
        memory = MemoryStore(constraints, model_hint="gpt-4o-mini")
        ledger = DummyLedger()
        llm = LLMClient()

        state, text, result = run_mistral_subquestion_tools(
            model="mistral-small-latest",
            memory=memory,
            step=1,
            sub_question="What is RAG?",
            constraints=constraints,
            ledger=ledger,
            llm=llm,
        )

        self.assertEqual(state["tool_calls"]["web_search"], 1)
        self.assertEqual(state["tool_calls"]["fetch_url"], 1)
        self.assertIn("mock_llm_enabled", state["errors"])
        self.assertIn("SUBQUESTION_DONE", text)
        self.assertEqual(result.prompt_tokens, 1)
        self.assertEqual(len(memory.chunks), 2)
        self.assertEqual(ledger.calls, [])

    def test_mock_mode_respects_zero_fetch_budget(self) -> None:
        constraints = ResearchConstraints(max_url_fetches_per_subq=0)
        memory = MemoryStore(constraints, model_hint="gpt-4o-mini")

        state, _, _ = run_mistral_subquestion_tools(
            model="mistral-small-latest",
            memory=memory,
            step=1,
            sub_question="What is RAG?",
            constraints=constraints,
        )

        self.assertEqual(state["tool_calls"]["web_search"], 1)
        self.assertEqual(state["tool_calls"]["fetch_url"], 0)
        self.assertEqual(len(memory.chunks), 1)

    def test_mock_mode_cost_abort_blocks_tool_calls(self) -> None:
        constraints = ResearchConstraints(max_url_fetches_per_subq=1)
        memory = MemoryStore(constraints, model_hint="gpt-4o-mini")

        state, _, _ = run_mistral_subquestion_tools(
            model="mistral-small-latest",
            memory=memory,
            step=1,
            sub_question="What is RAG?",
            constraints=constraints,
            cost_abort_fn=lambda: True,
        )

        self.assertEqual(state["tool_calls"]["web_search"], 0)
        self.assertEqual(state["tool_calls"]["fetch_url"], 0)
        self.assertEqual(len(memory.chunks), 0)


if __name__ == "__main__":
    unittest.main()
