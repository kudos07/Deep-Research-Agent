import unittest

from pydantic import ValidationError

from research_agent.config import ResearchConstraints


class ConfigTests(unittest.TestCase):
    def test_defaults_are_assignment_friendly(self) -> None:
        constraints = ResearchConstraints()
        self.assertEqual(constraints.max_context_tokens_per_llm_call, 2000)
        self.assertEqual(constraints.max_session_memory_tokens, 8000)
        self.assertEqual(constraints.max_search_results_per_subq, 3)

    def test_invalid_context_budget_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            ResearchConstraints(max_context_tokens_per_llm_call=100)

    def test_invalid_subquestion_cap_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            ResearchConstraints(max_sub_questions=99)

    def test_negative_cost_budget_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            ResearchConstraints(max_cost_usd_per_session=-1.0)


if __name__ == "__main__":
    unittest.main()
