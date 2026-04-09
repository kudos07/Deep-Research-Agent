import os
import unittest
from unittest.mock import patch

from research_agent.tools import SearchHit, _truthy_env, fetch_url_text, search_web


class ToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_env = {
            "RESEARCH_USE_MOCK_TOOLS": os.environ.get("RESEARCH_USE_MOCK_TOOLS"),
            "RESEARCH_AUTO_MOCK_ON_EMPTY": os.environ.get("RESEARCH_AUTO_MOCK_ON_EMPTY"),
        }

    def tearDown(self) -> None:
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_truthy_env_recognizes_enabled_values(self) -> None:
        os.environ["RESEARCH_USE_MOCK_TOOLS"] = "yes"
        self.assertTrue(_truthy_env("RESEARCH_USE_MOCK_TOOLS"))

    def test_search_web_uses_mock_hits_when_enabled(self) -> None:
        os.environ["RESEARCH_USE_MOCK_TOOLS"] = "1"
        hits = search_web("rag failure modes", max_results=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].href, "https://example.org/mock")

    def test_search_web_filters_empty_rows(self) -> None:
        os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)

        class FakeDDGS:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def text(self, query: str, max_results: int):
                return [
                    {"title": "", "href": "", "body": ""},
                    {"title": "Good", "href": "https://example.org", "body": "useful"},
                ]

        with patch("research_agent.tools.DDGS", FakeDDGS):
            hits = search_web("rag", max_results=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Good")

    def test_search_web_falls_back_to_mock_when_enabled_and_live_search_empty(self) -> None:
        os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)
        os.environ["RESEARCH_AUTO_MOCK_ON_EMPTY"] = "1"

        class EmptyDDGS:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def text(self, query: str, max_results: int):
                return []

        with patch("research_agent.tools.DDGS", EmptyDDGS):
            hits = search_web("rag", max_results=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].href, "https://example.org/mock")

    def test_fetch_url_text_rejects_non_http_urls(self) -> None:
        self.assertEqual(fetch_url_text("file:///tmp/test.txt"), "")

    def test_fetch_url_text_uses_mock_text_when_enabled(self) -> None:
        os.environ["RESEARCH_USE_MOCK_TOOLS"] = "1"
        text = fetch_url_text("https://example.org/mock")
        self.assertIn("Mock page extract", text)

    def test_fetch_url_text_strips_html_content(self) -> None:
        os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)

        class FakeResponse:
            headers = {"content-type": "text/html"}
            text = "<html><body><h1>Title</h1><script>bad()</script><p>Body text</p></body></html>"

            def raise_for_status(self) -> None:
                return None

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

        with patch("research_agent.tools.httpx.Client", FakeClient):
            text = fetch_url_text("https://example.org/page")

        self.assertIn("Title", text)
        self.assertIn("Body text", text)
        self.assertNotIn("bad()", text)

    def test_fetch_url_text_returns_empty_on_request_error(self) -> None:
        os.environ.pop("RESEARCH_USE_MOCK_TOOLS", None)

        class FailingClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def get(self, url: str):
                raise RuntimeError("network down")

        with patch("research_agent.tools.httpx.Client", FailingClient):
            text = fetch_url_text("https://example.org/page")

        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
