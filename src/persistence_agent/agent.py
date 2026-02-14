from __future__ import annotations

from dataclasses import dataclass
from operator import add
import re
from typing import Annotated, Any
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    user_message: str
    memory_hits: list[str]
    timeline: Annotated[list[str], add]
    stored_fact: NotRequired[str]
    response: NotRequired[str]


@dataclass
class AgentContext:
    user_id: str


_NAME_PATTERN = re.compile(r"\bmy name is\s+([a-zA-Z][a-zA-Z '-]{0,60})", re.IGNORECASE)
_LIKES_PATTERN = re.compile(r"\bi (?:like|love)\s+([a-zA-Z0-9 ,.'-]{1,80})", re.IGNORECASE)


def _memory_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "memories")


def _extract_fact(message: str) -> str | None:
    name_match = _NAME_PATTERN.search(message)
    if name_match:
        return f"name: {name_match.group(1).strip()}"

    likes_match = _LIKES_PATTERN.search(message)
    if likes_match:
        return f"likes: {likes_match.group(1).strip()}"

    return None


def load_memories(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    namespace = _memory_namespace(runtime.context.user_id)
    records = runtime.store.search(namespace, limit=10)
    known_facts = [
        item.value["fact"]
        for item in records
        if isinstance(item.value, dict) and isinstance(item.value.get("fact"), str)
    ]

    return {
        "memory_hits": known_facts[-3:],
        "timeline": [f"loaded_memories:{len(known_facts)}"],
    }


def remember_fact(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    fact = _extract_fact(state["user_message"])
    if not fact:
        return {"timeline": ["remembered_fact:0"]}

    runtime.store.put(
        _memory_namespace(runtime.context.user_id),
        str(uuid.uuid4()),
        {"fact": fact},
    )

    return {
        "stored_fact": fact,
        "timeline": ["remembered_fact:1"],
    }


def respond(state: AgentState) -> dict[str, Any]:
    memory_hits = list(state.get("memory_hits", []))
    stored_fact = state.get("stored_fact")
    if stored_fact and stored_fact not in memory_hits:
        memory_hits.append(stored_fact)

    memory_summary = " | ".join(memory_hits) or "no prior memory"
    return {
        "response": f"You said: {state['user_message']}\nMemory: {memory_summary}",
        "timeline": ["responded"],
    }


def build_graph(*, checkpointer: InMemorySaver | None = None, store: InMemoryStore | None = None):
    saver = checkpointer or InMemorySaver()
    memory_store = store or InMemoryStore()

    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("load_memories", load_memories)
    builder.add_node("remember_fact", remember_fact)
    builder.add_node("respond", respond)

    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "remember_fact")
    builder.add_edge("remember_fact", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile(checkpointer=saver, store=memory_store)
    return graph, saver, memory_store


def make_config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, dict[str, str]]:
    configurable: dict[str, str] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def run_turn(
    graph: Any,
    *,
    thread_id: str,
    user_id: str,
    message: str | None,
    checkpoint_id: str | None = None,
):
    config = make_config(thread_id, checkpoint_id)
    payload = None if message is None else {"user_message": message, "memory_hits": [], "timeline": []}
    return graph.invoke(payload, config=config, context=AgentContext(user_id=user_id))


def latest_state(graph: Any, thread_id: str, checkpoint_id: str | None = None):
    return graph.get_state(make_config(thread_id, checkpoint_id))


def state_history(graph: Any, thread_id: str):
    return list(graph.get_state_history(make_config(thread_id)))


def edit_state(
    graph: Any,
    *,
    thread_id: str,
    values: dict[str, Any],
    checkpoint_id: str | None = None,
    as_node: str | None = None,
):
    config = make_config(thread_id, checkpoint_id)
    if as_node is None:
        return graph.update_state(config, values)
    return graph.update_state(config, values, as_node=as_node)
