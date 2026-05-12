from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Any

import httpx
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from tqdm import tqdm

load_dotenv()

LLMS_INDEX_URL = "https://docs.claude.com/llms.txt"
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
FETCH_CONCURRENCY = 10
USER_AGENT = "anthropic-case-ingestion/0.1"

# llms.txt bullet: "- [Title](https://.../page.md): description"
LINK_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\((https?://[^\)]+\.md)\)")

HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT,
    strip_headers=True,
)
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def slugify(text: str) -> str:
    """Mimic MkDocs/Anthropic docs anchor generation: lowercase, hyphens."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def cite_url_for(source_url: str, anchor: str | None) -> str:
    """Build the human-facing URL with anchor, dropping the .md suffix."""
    base = source_url.removesuffix(".md")
    return f"{base}#{anchor}" if anchor else base


def header_path(headers: dict[str, str]) -> str:
    """Join headers in order: H1 > H2 > H3 > H4."""
    parts = [headers.get(f"Header {i}") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


def deepest_and_section_anchors(headers: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (deep_anchor, section_anchor).

    deep_anchor    = slug of the deepest header present (most specific link)
    section_anchor = slug of Header 2 (more stable, broader section)
    """
    deepest = None
    for i in range(4, 0, -1):
        h = headers.get(f"Header {i}")
        if h:
            deepest = h
            break
    deep = slugify(deepest) if deepest else None
    section = slugify(headers["Header 2"]) if headers.get("Header 2") else None
    return deep, section


async def fetch_index(client: httpx.AsyncClient) -> list[tuple[str, str]]:
    r = await client.get(LLMS_INDEX_URL)
    r.raise_for_status()
    seen, links = set(), []
    for line in r.text.splitlines():
        m = LINK_RE.match(line)
        if not m:
            continue
        title, url = m.group(1), m.group(2)
        if url not in seen:
            seen.add(url)
            links.append((title, url))
    return links


async def fetch_pages(links: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Returns list of (title, url, markdown)."""
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/markdown,*/*"}

    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        async def fetch_one(title: str, url: str):
            async with sem:
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    return (title, url, r.text)
                except Exception as e:
                    print(f"  ! failed {url}: {e}", file=sys.stderr)
                    return None

        results = []
        tasks = [fetch_one(t, u) for t, u in links]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="fetch"):
            r = await coro
            if r:
                results.append(r)
    return results


def chunk_pages(pages: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Returns list of dicts ready for embedding + insert."""
    rows: list[dict[str, Any]] = []
    for title, url, md in pages:
        # Step 1: split by header (returns Documents with header metadata)
        sections = md_splitter.split_text(md) or []
        # Fallback: pages with no headers — treat whole page as one section
        if not sections:
            sections = [Document(page_content=md, metadata={})]

        # Step 2: enforce size cap on each section
        for sec in sections:
            sub_chunks = size_splitter.split_text(sec.page_content)
            for piece in sub_chunks:
                piece = piece.strip()
                if len(piece) < 100:  # skip near-empty, low information
                    continue
                deep, section = deepest_and_section_anchors(sec.metadata)
                rows.append({
                    "url": url,
                    "title": title,
                    "headers": sec.metadata,
                    "chunk": piece,
                    "metadata": {
                        "cite_url": cite_url_for(url, deep),
                        "anchor_deep": deep,
                        "anchor_section": section,
                        "char_count": len(piece),
                    },
                })
    return rows


def embed_rows(rows: list[dict[str, Any]]) -> list[list[float]]:
    """Embed with title + header_path prepended for better retrieval signal.

    Without this, a chunk like "Use stream=True..." has no API context;
    with it, the embedding sees "Messages API > Streaming > ..." too.
    """
    client = OpenAI()
    out: list[list[float]] = []
    for i in tqdm(range(0, len(rows), EMBED_BATCH), desc="embed"):
        batch = rows[i : i + EMBED_BATCH]
        inputs = [
            f"{r['title']} — {header_path(r['headers'])}\n\n{r['chunk']}"
            for r in batch
        ]
        resp = client.embeddings.create(model=EMBED_MODEL, input=inputs)
        out.extend(d.embedding for d in resp.data)
    return out


def write_to_db(rows: list[dict[str, Any]], embeddings: list[list[float]], reset: bool) -> int:
    inserted = 0
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if reset:
                cur.execute("truncate table documents restart identity;")
                print("  truncated documents table")

            sql = """
                insert into documents (url, title, headers, chunk, embedding, metadata)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (url, chunk) do nothing
            """
            for r, emb in tqdm(list(zip(rows, embeddings)), desc="insert"):
                cur.execute(sql, (
                    r["url"], r["title"], Jsonb(r["headers"]),
                    r["chunk"], emb, Jsonb(r["metadata"]),
                ))
                inserted += cur.rowcount
        conn.commit()
    return inserted

async def main(limit: int | None, reset: bool) -> None:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"}
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        links = await fetch_index(client)
    print(f"-> {len(links)} pages in index")

    if limit:
        links = links[:limit]
        print(f"-> limited to {len(links)}")

    pages = await fetch_pages(links)
    print(f"-> fetched {len(pages)}")

    rows = chunk_pages(pages)
    print(f"-> {len(rows)} chunks (avg {sum(r['metadata']['char_count'] for r in rows) // max(len(rows), 1)} chars)")
    if not rows:
        return

    embeddings = embed_rows(rows)
    inserted = write_to_db(rows, embeddings, reset=reset)
    print(f"-> inserted {inserted}/{len(rows)} new rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    asyncio.run(main(limit=args.limit, reset=args.reset))