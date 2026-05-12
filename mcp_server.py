from __future__ import annotations

import sys
import traceback
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from retrieval import retrieve

load_dotenv()
mcp = FastMCP("anthropic-docs")
USER_AGENT = "anthropic-docs-mcp/0.1"
FETCH_TIMEOUT = 15


def _header_path(headers: dict) -> str:
    parts = [headers.get(f"Header {i}") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


@mcp.tool()
def search_docs(
    query: str = Field(..., description="Natural-language question or keywords to search for in the Anthropic Claude documentation."),
    top_k: int = Field(5, ge=1, le=10, description="How many chunks to return. Default 5 is a good balance for most questions."),
    ) -> list[dict]:
    """
    Search the indexed Anthropic Claude documentation.

    Returns the top matching chunks ranked by semantic similarity. Each chunk
    includes the source URL (with a deep-link anchor to the exact section),
    the page title and header path, and the chunk text.

    Use this for any question about the Claude API, SDKs, agent skills,
    or platform features. Prefer this over guessing from prior knowledge —
    the docs change frequently.
    """
    chunks = retrieve(query, top_k=top_k)
    return [
        {
            "source": c.cite_url,
            "title": f"{c.title} — {_header_path(c.headers)}" if _header_path(c.headers) else c.title,
            "content": c.chunk,
            "similarity": round(c.similarity, 4),
        }
        for c in chunks
    ]


@mcp.tool()
def get_full_page(
    url: str = Field(..., description="Canonical URL of an Anthropic docs page. Must end in `.md` (e.g. https://docs.claude.com/en/api/messages.md). Get URLs from search_docs results."),
) -> str:
    """
    Fetch the full markdown content of a documentation page.

    Use this after search_docs when the top chunks don't contain enough
    context and you need the full page to answer accurately. Returns raw
    markdown.
    """
    if not url.endswith(".md"):
        return f"Error: URL must point to a .md file. Got: {url}"

    headers = {"User-Agent": USER_AGENT, "Accept": "text/markdown,*/*"}
    try:
        r = httpx.get(url, headers=headers, timeout=FETCH_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.text

    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        traceback.print_exc(file=sys.stderr) # just to help me debug 
        raise