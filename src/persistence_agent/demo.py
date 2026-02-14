from __future__ import annotations

from persistence_agent.agent import AgentContext, build_graph, edit_state, run_turn, state_history


def main() -> None:
    graph, _, _ = build_graph()

    print("== Thread 1: initial turn ==")
    first = run_turn(graph, thread_id="thread-1", user_id="user-1", message="my name is Jon")
    print(first["response"])

    print("\n== Thread 1: follow-up turn ==")
    second = run_turn(graph, thread_id="thread-1", user_id="user-1", message="i like pizza")
    print(second["response"])

    print("\n== Thread 2: same user, memory shared via store ==")
    cross_thread = run_turn(
        graph,
        thread_id="thread-2",
        user_id="user-1",
        message="what do you remember about me?",
    )
    print(cross_thread["response"])

    print("\n== Checkpoint history for thread-1 ==")
    history = state_history(graph, "thread-1")
    print(f"checkpoint_count={len(history)}")

    if len(history) > 1:
        selected = history[1]
        print("\n== Replay from a prior checkpoint ==")
        replayed = graph.invoke(None, config=selected.config, context=AgentContext(user_id="user-1"))
        print(replayed["response"])

    print("\n== Fork by editing state ==")
    new_config = edit_state(
        graph,
        thread_id="thread-1",
        values={"user_message": "i love ramen"},
        as_node="load_memories",
    )
    forked = graph.invoke(None, config=new_config, context=AgentContext(user_id="user-1"))
    print(forked["response"])


if __name__ == "__main__":
    main()
