import unittest

from research_agent.config import ResearchConstraints
from research_agent.memory import MemoryStore
from research_agent.token_utils import estimate_tokens


class MemoryStoreTests(unittest.TestCase):
    def test_maybe_compact_merges_old_observations_into_summary(self) -> None:
        constraints = ResearchConstraints(
            max_session_memory_tokens=2000,
            summary_target_tokens=200,
        )
        store = MemoryStore(constraints, model_hint="gpt-4o-mini")

        for step in range(1, 4):
            store.add_observation(
                text=("alpha beta gamma " * 260).strip(),
                source=f"obs:{step}",
                step=step,
            )

        before_tokens = store.total_tokens()
        logs = store.maybe_compact(lambda prompt: "summary bullet\n" * 10)

        self.assertGreater(before_tokens, constraints.max_session_memory_tokens)
        self.assertTrue(logs)
        self.assertLess(store.total_tokens(), before_tokens)
        self.assertTrue(any(chunk.kind == "summary" for chunk in store.chunks))
        self.assertTrue(any(chunk.kind == "observation" for chunk in store.chunks))

    def test_select_context_picks_relevant_chunks_within_budget(self) -> None:
        constraints = ResearchConstraints()
        store = MemoryStore(constraints, model_hint="gpt-4o-mini")

        store.add_observation(
            text="RAG retrieval ranking chunking reranking improves grounding.",
            source="relevant",
            step=1,
        )
        store.add_observation(
            text="Tropical fish habitat and coral reef migration patterns.",
            source="irrelevant",
            step=2,
        )

        budget_tokens = estimate_tokens(
            "## Retrieved memory\n### relevant (step 1, observation)\n"
            "RAG retrieval ranking chunking reranking improves grounding.\n\n",
            "gpt-4o-mini",
        ) + 5

        blob, picked_ids = store.select_context(
            "What retrieval improvements help a naive RAG system?",
            budget_tokens=budget_tokens,
        )

        picked_sources = {chunk.source for chunk in store.get_by_ids(picked_ids)}
        self.assertIn("relevant", picked_sources)
        self.assertNotIn("irrelevant", picked_sources)
        self.assertLessEqual(estimate_tokens(blob, "gpt-4o-mini"), budget_tokens)


if __name__ == "__main__":
    unittest.main()
