import os
import unittest

from research_agent.agent import run_research
from research_agent.config import ResearchConstraints


class RunResearchTests(unittest.TestCase):
    def test_mock_tools_mode_runs_without_api_keys_and_returns_answer(self) -> None:
        saved_env = {
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        }
        try:
            for key in saved_env:
                os.environ.pop(key, None)

            result = run_research(
                "In two parts: define RAG and name one naive failure mode with a mitigation.",
                constraints=ResearchConstraints(
                    max_context_tokens_per_llm_call=2000,
                    max_session_memory_tokens=8000,
                    max_sub_questions=4,
                ),
                use_mock_tools=True,
            )
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertTrue(result["sub_questions"])
        self.assertTrue(result["retrieval"]["selected_chunk_ids"])
        self.assertIn("Executive answer", result["answer"])
        self.assertGreaterEqual(len(result["execution"]["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
