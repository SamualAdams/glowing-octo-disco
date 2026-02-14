# LangGraph Persistence Skeleton

Minimal agent skeleton for testing LangGraph persistence behavior before expanding into a full implementation.

## What this includes

- Thread-level checkpointing with `InMemorySaver`
- Cross-thread memory with `InMemoryStore`
- Optional OpenAI-backed response generation (opt-in per graph build)
- A small graph with 3 nodes:
  - `load_memories`
  - `remember_fact`
  - `respond`
- Helpers for:
  - `get_state`
  - `get_state_history`
  - replay from `checkpoint_id`
  - `update_state` forks
- Pytest coverage for core persistence flows

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

- `OPENAI_API_KEY` for model access
- optional `OPENAI_MODEL` override
- optional LangSmith variables for tracing

## Run the demo (uv)

```bash
PYTHONPATH=src uv run --env-file .env python -m persistence_agent.demo
```

`demo.py` now builds the graph with `use_openai_from_env=True`, so if `OPENAI_API_KEY` is set it will use OpenAI for the `respond` node.
If no key is set, it falls back to deterministic local response text.

## Run tests (uv)

```bash
PYTHONPATH=src uv run --env-file .env pytest
```

## Notes

This is intentionally a skeleton for pairing and PR review. It uses in-memory checkpoint/store implementations for local iteration only.
