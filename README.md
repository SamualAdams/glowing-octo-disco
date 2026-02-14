# LangGraph Persistence Skeleton

Minimal agent skeleton for testing LangGraph persistence behavior before expanding into a full implementation.

## What this includes

- Thread-level checkpointing with `InMemorySaver`
- Cross-thread memory with `InMemoryStore`
- An LLM-backed ReAct agent loop (OpenAI chat model)
- Tool-based memory operations:
  - `remember_fact`
  - `search_memories`
- Helpers for:
  - `get_state` / `latest_state`
  - `get_state_history` / `state_history`
  - replay from `checkpoint_id`
  - `update_state` forks
- Integration tests for OpenAI connectivity + persistence flows

## Install (uv)

```bash
uv sync --extra dev
```

## Docker prerequisite

Install and run Docker Desktop (or another Docker daemon) before running tests. The test suite now starts and stops a local Postgres container automatically.

## Install (pip fallback)

```bash
python3 -m pip install -r requirements.txt
```

## Environment variables

```bash
cp .env.example .env
```

Set your credentials in `.env`:

- `PERSISTENCE_AGENT_OPENAI_API_KEY` for model access
- optional `OPENAI_MODEL` override
- optional `OPENAI_TIMEOUT_SECONDS` override
- local Postgres defaults:
  - `DATABASE_URL=postgresql://app:app@127.0.0.1:5442/app`
  - `PGHOST=127.0.0.1`
  - `PGPORT=5442`
  - `PGDATABASE=app`
  - `PGUSER=app`
  - `PGPASSWORD=app`
- optional LangSmith variables for tracing

## Run the demo (uv)

```bash
PYTHONPATH=src uv run --env-file .env python -m persistence_agent.demo
```

## Run tests (uv)

```bash
PYTHONPATH=src uv run --env-file .env pytest
```

`pytest` brings up `postgres` via Docker Compose at session start, waits for readiness, runs DB connectivity tests, and tears the container down at session end.

If tests fail with Docker daemon errors, start Docker Desktop and rerun.

## Notes

This is intentionally a skeleton for pairing and PR review. It uses in-memory checkpoint/store implementations for local iteration only.
