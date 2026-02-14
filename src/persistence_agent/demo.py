from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from persistence_agent.agent import AgentContext, build_graph, edit_state, run_turn, state_history


def _latest_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                return msg.content
            return str(msg.content)
    return ""


def main() -> None:
    graph, _, _ = build_graph()

    print("== Thread 1: initial turn ==")
    first = run_turn(
        graph,
        thread_id="thread-1",
        user_id="user-1",
        message="My name is Jon. Please remember this as a durable fact.",
    )
    print(first["response"])

    print("\n== Thread 1: follow-up turn ==")
    second = run_turn(
        graph,
        thread_id="thread-1",
        user_id="user-1",
        message="I like pizza. Save that preference too.",
    )
    print(second["response"])
    print(f"memory_events={len(second['memory_events'])}")

    print("\n== Thread 2: same user, memory shared via store ==")
    cross_thread = run_turn(
        graph,
        thread_id="thread-2",
        user_id="user-1",
        message="What do you remember about me? Use memory search.",
    )
    print(cross_thread["response"])

    print("\n== Checkpoint history for thread-1 ==")
    history = state_history(graph, "thread-1")
    print(f"checkpoint_count={len(history)}")

    if len(history) > 1:
        selected = history[1]
        print("\n== Replay from a prior checkpoint ==")
        replayed = graph.invoke(None, config=selected.config, context=AgentContext(user_id="user-1"))
        print(_latest_ai_text(replayed.get("messages", [])))

    print("\n== Fork by editing state ==")
    new_config = edit_state(
        graph,
        thread_id="thread-1",
        values={"messages": [HumanMessage(content="I love ramen. Remember that too.")]},
        as_node="model",
    )
    fork_checkpoint_id = new_config["configurable"].get("checkpoint_id")
    print(f"fork_checkpoint_id={fork_checkpoint_id}")
    forked = run_turn(
        graph,
        thread_id="thread-1",
        user_id="user-1",
        checkpoint_id=fork_checkpoint_id,
        message="What preferences do you remember?",
    )
    print(forked["response"])


if __name__ == "__main__":
    main()
