import os
import unittest

from fastapi.testclient import TestClient

from research_agent.api import app


class ApiTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        client = TestClient(app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_research_endpoint_allows_mock_mode_without_api_key(self) -> None:
        saved_env = {
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        }
        try:
            for key in saved_env:
                os.environ.pop(key, None)

            client = TestClient(app)
            response = client.post(
                "/research",
                json={
                    "question": "Explain RAG and name one common failure mode.",
                    "mock_tools": True,
                },
            )
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertTrue(data["retrieval"]["selected_chunk_ids"])

    def test_research_endpoint_requires_key_without_mock_mode(self) -> None:
        saved_env = {
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        }
        try:
            for key in saved_env:
                os.environ.pop(key, None)

            client = TestClient(app)
            response = client.post(
                "/research",
                json={
                    "question": "Explain RAG and name one common failure mode.",
                    "mock_tools": False,
                },
            )
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(response.status_code, 503)

    def test_research_endpoint_validates_short_question(self) -> None:
        client = TestClient(app)
        response = client.post("/research", json={"question": "hi", "mock_tools": True})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
