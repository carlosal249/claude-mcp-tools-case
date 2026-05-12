from __future__ import annotations

import os
import json
import anthropic
from fastapi import FastAPI
from dotenv import load_dotenv
from typing import AsyncIterator
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

# Local files
from prompts import SYSTEM_PROMPT
from retrieval import RetrievedChunk, retrieve

load_dotenv()

MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-4-5") 
MAX_TOKENS = 1024
TOP_K = 5

app = FastAPI(title="Anthropic Docs RAG")
_anthropic: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic()
    return _anthropic

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)

def _header_path(headers: dict) -> str:
    parts = [headers.get(f"Header {i}") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


def _to_search_results(chunks: list[RetrievedChunk]) -> list[dict]:
    """Wrap retrieved chunks as Search Result blocks for the Citations API.

    Search Result blocks are designed for RAG — citations come back with
    `source` and `title` already populated, no manual index-to-URL mapping.
    """
    blocks: list[dict] = []
    for c in chunks:
        path = _header_path(c.headers)
        title = f"{c.title} — {path}" if path else c.title
        blocks.append({
            "type": "search_result",
            "source": c.cite_url,
            "title": title,
            "content": [{"type": "text", "text": c.chunk}],
            "citations": {"enabled": True},
        })
    return blocks


def _sse(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

async def _stream_answer(question: str, top_k: int) -> AsyncIterator[str]:
    chunks = retrieve(question, top_k=top_k)

    # Surface the retrieved sources up front (useful for debugging + the eval).
    yield _sse("retrieval", {
        "chunks": [
            {
                "url": c.url,
                "cite_url": c.cite_url,
                "title": c.title,
                "header_path": _header_path(c.headers),
                "similarity": round(c.similarity, 4),
            }
            for c in chunks
        ]
    })

    if not chunks:
        yield _sse("text", {"delta": "I don't have information about that in the indexed documentation."})
        yield _sse("done", {})
        return

    search_blocks = _to_search_results(chunks)
    user_content = search_blocks + [{"type": "text", "text": question}]

    # The Anthropic SDK's streaming context manager parses the event stream
    # for us. We re-emit only what the client cares about.
    with _client().messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", None)

            if etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)

                if dtype == "text_delta":
                    yield _sse("text", {"delta": delta.text})

                elif dtype == "citations_delta":
                    # Citation references one of our Search Result blocks.
                    # `source` and `title` come straight from the block we sent.
                    cit = delta.citation
                    yield _sse("citation", {
                        "source": getattr(cit, "source", None),
                        "title": getattr(cit, "title", None),
                        "cited_text": getattr(cit, "cited_text", None),
                    })

            elif etype == "message_stop":
                break

    yield _sse("done", {})

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.post("/ask")
async def ask(req: AskRequest):
    return StreamingResponse(
        _stream_answer(req.question, req.top_k),
        media_type="text/event-stream",
    )