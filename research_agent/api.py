"""FastAPI surface for n8n (or Dify HTTP tool) orchestration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_agent.agent import run_research
from research_agent.config import ResearchConstraints

load_dotenv()

app = FastAPI(title="G3 Research Agent API", version="0.2.0")

# Lets the static site (or any local dev server) call POST /research from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_WEBSITE_DIR = Path(__file__).resolve().parent.parent / "website"
if _WEBSITE_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEBSITE_DIR), html=True), name="ui")


class ResearchBody(BaseModel):
    question: str = Field(min_length=3)
    mock_tools: bool = False
    max_context_tokens_per_llm_call: int | None = None
    max_session_memory_tokens: int | None = None
    max_cost_usd_per_session: float | None = None


@app.get("/")
def root():
    if _WEBSITE_DIR.is_dir():
        return RedirectResponse(url="/ui/", status_code=302)
    return {"ok": True, "message": "G3 Research Agent API", "docs": "/docs", "research": "POST /research"}


@app.get("/health")
def health():
    return {"ok": True, "llm_provider": os.getenv("LLM_PROVIDER", "(auto: MISTRAL_API_KEY preferred)")}


@app.post("/research")
def research(body: ResearchBody):
    if (
        not body.mock_tools
        and not os.getenv("MISTRAL_API_KEY")
        and not os.getenv("OPENAI_API_KEY")
        and not os.getenv("ANTHROPIC_API_KEY")
    ):
        raise HTTPException(status_code=503, detail="No LLM API key configured (set MISTRAL_API_KEY).")
    c = ResearchConstraints()
    if body.max_context_tokens_per_llm_call is not None:
        c.max_context_tokens_per_llm_call = body.max_context_tokens_per_llm_call
    if body.max_session_memory_tokens is not None:
        c.max_session_memory_tokens = body.max_session_memory_tokens
    if body.max_cost_usd_per_session is not None:
        c.max_cost_usd_per_session = body.max_cost_usd_per_session
    try:
        return run_research(body.question, constraints=c, use_mock_tools=body.mock_tools)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
