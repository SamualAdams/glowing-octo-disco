# LangGraph Persistence Skeleton

Minimal agent skeleton for testing LangGraph persistence behavior before expanding into a full implementation.

## What this includes

- Thread-level checkpointing with `InMemorySaver`
- Cross-thread memory with `InMemoryStore`
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

## Run the demo (uv)

```bash
uv run python -m persistence_agent.demo
```

## Run tests (uv)

```bash
uv run pytest
```

## Notes

This is intentionally a skeleton for pairing and PR review. It uses in-memory checkpoint/store implementations for local iteration only.
