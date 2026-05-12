from __future__ import annotations

import os
import psycopg
from openai import OpenAI
from dataclasses import dataclass
from pgvector.psycopg import register_vector

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5

_openai: OpenAI | None = None


def _client() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


@dataclass
class RetrievedChunk:
    url: str            # canonical .md source
    title: str          # title of the extracted topic
    headers: dict       # {"Header 1": "...", "Header 2": "...", ...}
    chunk: str          # the actual text of the chunk
    cite_url: str       # human URL with anchor for the citation
    similarity: float   # Similarity -> (Cosine)


def embed_query(question: str) -> list[float]:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=[question])
    return resp.data[0].embedding


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    embedding = embed_query(question)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "select url, title, headers, chunk, metadata, similarity "
                "from match_documents(%s::vector, %s)",
                (embedding, top_k),
            )
            rows = cur.fetchall()

    results: list[RetrievedChunk] = []
    for url, title, headers, chunk, metadata, similarity in rows:
        results.append(RetrievedChunk(
            url=url,
            title=title,
            headers=headers or {},
            chunk=chunk,
            cite_url=(metadata or {}).get("cite_url", url),
            similarity=float(similarity),
        ))
    return results