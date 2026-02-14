from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from persistence_agent.agent import (
    OPENAI_API_KEY_ENV,
    AgentContext,
    build_graph,
    edit_state,
    latest_state,
    run_turn,
    state_history,
)

pytestmark = pytest.mark.integration


def _require_openai_key() -> None:
    if not os.getenv(OPENAI_API_KEY_ENV):
        pytest.skip(f"{OPENAI_API_KEY_ENV} is not set.")


def test_checkpoints_are_recorded_per_thread() -> None:
    _require_openai_key()
    graph, _, _ = build_graph()

    result = run_turn(
        graph,
        thread_id="t-1",
        user_id="u-1",
        message="My name is Jon. Remember this as a durable memory.",
    )

    latest = latest_state(graph, "t-1")
    history = state_history(graph, "t-1")

    assert result["response"].strip()
    assert isinstance(latest.values.get("messages"), list)
    assert any(isinstance(msg, AIMessage) for msg in latest.values["messages"])
    assert latest.next == ()
    assert len(history) >= 1


def test_store_persists_across_threads_for_same_user() -> None:
    _require_openai_key()
    graph, _, store = build_graph()

    run_turn(
        graph,
        thread_id="t-1",
        user_id="u-1",
        message="My name is Jon. Save this as kind=name.",
    )
    stored = store.search(("u-1", "memories"), limit=20)
    assert any(
        isinstance(item.value, dict)
        and item.value.get("kind") == "name"
        and isinstance(item.value.get("value"), str)
        and "jon" in item.value["value"].lower()
        for item in stored
    )

    output = run_turn(
        graph,
        thread_id="t-2",
        user_id="u-1",
        message="What is my name? Use memory search before responding.",
    )

    assert "jon" in output["response"].lower()


def test_replay_and_update_state_create_forks() -> None:
    _require_openai_key()
    graph, _, _ = build_graph()

    run_turn(
        graph,
        thread_id="t-1",
        user_id="u-1",
        message="I like pizza. Store this preference.",
    )
    history = state_history(graph, "t-1")
    assert len(history) >= 1

    selected = history[0]
    replayed = graph.invoke(None, config=selected.config, context=AgentContext(user_id="u-1"))
    replayed_messages = replayed.get("messages", [])
    assert any(isinstance(msg, AIMessage) for msg in replayed_messages)

    new_config = edit_state(
        graph,
        thread_id="t-1",
        values={"messages": [HumanMessage(content="I like ramen. Save it as a preference.")]},
        as_node="model",
    )
    assert isinstance(new_config, dict)
    assert "configurable" in new_config
    assert new_config["configurable"].get("checkpoint_id")
    forked = graph.invoke(None, config=new_config, context=AgentContext(user_id="u-1"))
    ai_messages = [msg for msg in forked.get("messages", []) if isinstance(msg, AIMessage)]
    assert ai_messages
    final_text = ai_messages[-1].content if isinstance(ai_messages[-1].content, str) else str(ai_messages[-1].content)
    assert final_text.strip()
