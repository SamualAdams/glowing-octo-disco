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
- optional LangSmith variables for tracing

## Run the demo (uv)

```bash
PYTHONPATH=src uv run --env-file .env python -m persistence_agent.demo
```

## Run tests (uv)

```bash
PYTHONPATH=src uv run --env-file .env pytest
```

## Notes

This is intentionally a skeleton for pairing and PR review. It uses in-memory checkpoint/store implementations for local iteration only.
