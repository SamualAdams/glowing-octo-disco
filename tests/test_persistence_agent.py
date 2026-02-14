from __future__ import annotations

from persistence_agent.agent import AgentContext, build_graph, edit_state, latest_state, run_turn, state_history


def test_checkpoints_are_recorded_per_thread() -> None:
    graph, _, _ = build_graph()

    run_turn(graph, thread_id="t-1", user_id="u-1", message="my name is Jon")

    latest = latest_state(graph, "t-1")
    history = state_history(graph, "t-1")

    assert latest.values["response"].startswith("You said: my name is Jon")
    assert latest.next == ()
    assert len(history) >= 4


def test_store_persists_across_threads_for_same_user() -> None:
    graph, _, _ = build_graph()

    run_turn(graph, thread_id="t-1", user_id="u-1", message="my name is Jon")
    output = run_turn(graph, thread_id="t-2", user_id="u-1", message="what do you remember?")

    assert "name: Jon" in output["response"]


def test_replay_and_update_state_create_forks() -> None:
    graph, _, _ = build_graph()

    run_turn(graph, thread_id="t-1", user_id="u-1", message="i like pizza")
    history = state_history(graph, "t-1")
    assert len(history) >= 2

    selected = history[1]
    replayed = graph.invoke(None, config=selected.config, context=AgentContext(user_id="u-1"))
    assert replayed["response"].startswith("You said:")

    new_config = edit_state(
        graph,
        thread_id="t-1",
        values={"user_message": "i like ramen"},
        as_node="load_memories",
    )
    forked = graph.invoke(None, config=new_config, context=AgentContext(user_id="u-1"))
    assert "ramen" in forked["response"]


def test_injected_llm_responder_is_used() -> None:
    def fake_llm(*, user_message: str, memory_summary: str) -> str:
        return f"LLM::{user_message}::{memory_summary}"

    graph, _, _ = build_graph(llm_responder=fake_llm)
    output = run_turn(graph, thread_id="t-llm", user_id="u-1", message="my name is Jon")

    assert output["response"].startswith("LLM::my name is Jon::")
