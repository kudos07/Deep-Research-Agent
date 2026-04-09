"""Web search and lightweight page fetch (best-effort; network may fail)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore


@dataclass
class SearchHit:
    title: str
    href: str
    body: str


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _mock_hits(query: str, max_results: int) -> list[SearchHit]:
    """
    Deterministic offline snippets for demos/CI.

    Enable with RESEARCH_USE_MOCK_TOOLS=1.
    """
    q = query.strip()
    return [
        SearchHit(
            title="Mock knowledge card (offline demo)",
            href="https://example.org/mock",
            body=(
                f"Mock evidence snippet for query: {q}. "
                "RAG combines retrieval (fetching relevant documents) with generation (an LLM), "
                "but naive chunking can yield missing context; reranking and better chunking help."
            ),
        )
    ][:max_results]


def search_web(query: str, max_results: int = 3, timeout_s: float = 20.0) -> list[SearchHit]:
    if _truthy_env("RESEARCH_USE_MOCK_TOOLS"):
        return _mock_hits(query, max_results)
    try:
        ddgs = DDGS(timeout=timeout_s)
        raw = list(ddgs.text(query, max_results=max_results))
    except Exception:
        raw = []
    if not raw and _truthy_env("RESEARCH_AUTO_MOCK_ON_EMPTY"):
        return _mock_hits(query, max_results)
    hits: list[SearchHit] = []
    for row in raw:
        title = str(row.get("title") or "").strip()
        href = str(row.get("href") or "").strip()
        body = str(row.get("body") or "").strip()
        if not body and not title:
            continue
        hits.append(SearchHit(title=title, href=href, body=body))
    return hits


_WS = re.compile(r"\s+")


def fetch_url_text(url: str, max_chars: int = 12000, timeout_s: float = 20.0) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    if _truthy_env("RESEARCH_USE_MOCK_TOOLS"):
        return (
            "Mock page extract. This is not a live webpage. "
            "It exists so the agent can demonstrate fetch->memory->budgeted synthesis without brittle scraping."
        )[:max_chars]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/0.1; +student-demo)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
            r = client.get(url)
            r.raise_for_status()
            ct = (r.headers.get("content-type") or "").lower()
            if "text/html" not in ct and "application/xhtml" not in ct:
                # Plain text-ish
                text = r.text[:max_chars]
                return _WS.sub(" ", text).strip()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = _WS.sub(" ", text).strip()
            return text[:max_chars]
    except Exception:
        return ""
