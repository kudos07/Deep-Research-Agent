import json
import os
import unittest

from research_agent.llm import LLMClient, LLMResult


class LLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_env = {
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
            "LLM_PRICE_INPUT_PER_1K": os.environ.get("LLM_PRICE_INPUT_PER_1K"),
            "LLM_PRICE_OUTPUT_PER_1K": os.environ.get("LLM_PRICE_OUTPUT_PER_1K"),
            "RESEARCH_USE_MOCK_LLM": os.environ.get("RESEARCH_USE_MOCK_LLM"),
        }

    def tearDown(self) -> None:
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_resolve_provider_prefers_explicit_provider(self) -> None:
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["MISTRAL_API_KEY"] = "x"
        client = LLMClient()
        self.assertEqual(client.provider, "openai")

    def test_resolve_provider_defaults_to_mistral_when_key_present(self) -> None:
        os.environ.pop("LLM_PROVIDER", None)
        os.environ["MISTRAL_API_KEY"] = "x"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        client = LLMClient()
        self.assertEqual(client.provider, "mistral")

    def test_estimate_cost_uses_price_hints(self) -> None:
        os.environ["LLM_PRICE_INPUT_PER_1K"] = "0.5"
        os.environ["LLM_PRICE_OUTPUT_PER_1K"] = "1.5"
        client = LLMClient()
        result = LLMResult(text="x", prompt_tokens=1000, completion_tokens=2000)
        self.assertAlmostEqual(client.estimate_cost_usd(result), 3.5)

    def test_mock_complete_returns_json_plan_when_requested(self) -> None:
        client = LLMClient()
        result = client._mock_complete(
            "Return strict JSON",
            "User question:\nWhat is RAG?\n",
            200,
            json_mode=True,
        )
        payload = json.loads(result.text)
        self.assertIn("sub_questions", payload)
        self.assertGreaterEqual(len(payload["sub_questions"]), 2)

    def test_mock_complete_synthesizes_answer_from_retrieved_memory(self) -> None:
        client = LLMClient()
        user = (
            "Original question:\nWhat is RAG?\n\n"
            "## Retrieved memory\n"
            "### search_hit:1.1.1 (step 1, observation)\n"
            "RAG improves grounding with retrieval and reranking.\n\n"
            "Produce the final answer now."
        )
        result = client._mock_complete(
            "Answer using ONLY the provided retrieved memory.",
            user,
            300,
            json_mode=False,
        )
        self.assertIn("Executive answer", result.text)
        self.assertIn("Evidence", result.text)
        self.assertNotIn("Produce the final answer now", result.text)

    def test_mock_complete_summarizer_keeps_compact_lines(self) -> None:
        client = LLMClient()
        user = "\n".join(f"line {i}" for i in range(30))
        result = client._mock_complete(
            "You compress research notes. Be faithful; do not invent facts.",
            user,
            200,
            json_mode=False,
        )
        self.assertIn("line 0", result.text)
        self.assertNotIn("line 29", result.text)

    def test_loads_json_object_handles_markdown_fence(self) -> None:
        text = '```json\n{"sub_questions":[{"id":1,"question":"Q","focus":"F"}]}\n```'
        data = LLMClient._loads_json_object(text)
        self.assertEqual(data["sub_questions"][0]["question"], "Q")


if __name__ == "__main__":
    unittest.main()
