# Anthropic Docs RAG

A small RAG over `docs.claude.com`, exposed as a FastAPI endpoint and an MCP server.
Built as a technical case.

## Architecture

```
                    ┌────────────────┐
                    │  ingestion.py  │ 
                    │  llms.txt → md │
                    │  → chunks      │
                    │  → embeddings  │
                    └────────┬───────┘
                             ▼
                    ┌────────────────┐
                    │  Supabase      │
                    │  + pgvector    │
                    └────────┬───────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌────────────────┐   ┌────────────────┐
          │  retrieval.py  │   │  retrieval.py  │
          └────────┬───────┘   └────────┬───────┘
                   ▼                    ▼
          ┌────────────────┐   ┌────────────────┐
          │  FastAPI       │   │  MCP server    │
          │  /ask (SSE)    │   │  (stdio)       │
          │  + Citations   │   │  search_docs   │
          │    API         │   │  get_full_page │
          └────────────────┘   └────────────────┘
                   ▲                    ▲
                   │                    │
              eval.py             Claude Desktop
```

Both the `/ask` endpoint and the MCP server share the same retrieval layer
(`retrieval.py`) and Supabase database. They are independent processes —
the MCP does not call `/ask` over HTTP.

- **Ingestion** (`ingestion.py`): parses `llms.txt`, fetches individual `.md`
  pages, splits with `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`,
  embeds with OpenAI `text-embedding-3-small`, stores in Supabase with
  `header_path` prepended at embed time for better retrieval signal.
- **`/ask`** (`app.py`): FastAPI endpoint that retrieves top-5 chunks, wraps
  them as **Search Result blocks** (the RAG-native variant of the Citations API),
  and streams the answer over SSE with `text` / `citation` / `retrieval` events.
- **MCP server** (`mcp_server.py`): two tools — `search_docs(query, top_k)` and
  `get_full_page(url)` — connectable to any MCP client. Schemas defined with
  Pydantic; runs via stdio.
- **Eval** (`eval.py`): 12 hand-written Q/A pairs (5 factual, 3 synthesis,
  4 adversarial). Three independent metrics: precision@5 (deterministic),
  refusal accuracy (regex), LLM-as-judge (Haiku scoring faithfulness, relevance,
  citation correctness 1–5).

## Setup

### 1. Supabase

Create a project at [supabase.com](https://supabase.com). In the SQL Editor, run
the contents of `schema.sql`. This creates the `vector` extension, the
`documents` table with an HNSW index, and the `match_documents` RPC.

### 2. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres
```

URL-encode any special characters in the password (`@` → `%40`, etc.).

### 3. Ingestion

```bash
python ingestion.py --limit 5    # smoke test (~30 s)
python ingestion.py --reset      # full ingest (~5 min, ~$0.10)
```

`--reset` truncates the table first. Re-runs are idempotent
(`ON CONFLICT DO NOTHING` on `unique(url, chunk)`).

## Running

### FastAPI `/ask`

```bash
python -m uvicorn app:app --port 8000
```

Test:

```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Claude load PDF processing skills?"}'
```

Response is an SSE stream with four event types: `retrieval` (top-k chunks
with similarity scores), `text` (deltas), `citation` (source + cited text),
`done`.

### MCP server in Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) and add:

```json
{
  "mcpServers": {
    "anthropic-docs": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart Claude Desktop (Cmd+Q, then reopen). Ask something like
*"Search the Anthropic docs for how PDF processing works"* — the tool call
will appear in the chat with native citation rendering.

For local debugging without Claude Desktop:

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

### Eval

With `/ask` running:

```bash
python eval.py
```

Generates `eval_results.json` (raw) and `eval_report.md` (summary + per-question
table + automatic failure analysis).

## Files

| File | Purpose |
|---|---|
| `ingestion.py` | One-off pipeline: docs → Supabase |
| `schema.sql` | DDL + `match_documents` RPC |
| `retrieval.py` | Shared by `/ask` and MCP |
| `app.py` | FastAPI `/ask` with Citations API |
| `prompts.py` | System prompt (XML-tagged, with few-shot slots) |
| `mcp_server.py` | MCP server, two tools |
| `eval.py` | 3-metric evaluation script |
| `eval_dataset.json` | 12 Q/A pairs |
