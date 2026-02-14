from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Literal
import uuid

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore


@dataclass
class AgentContext:
    user_id: str


OPENAI_API_KEY_ENV = "PERSISTENCE_AGENT_OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0


def _memory_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "memories")


def _build_model(model_name: str | None = None) -> ChatOpenAI:
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{OPENAI_API_KEY_ENV} is required to build the LLM-backed graph.")

    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_OPENAI_TIMEOUT_SECONDS)))
    return ChatOpenAI(
        model=model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=api_key,
        temperature=0,
        timeout=timeout,
        max_retries=1,
    )


def _dedupe_key(kind: str, value: str) -> tuple[str, str]:
    return (kind.strip().lower(), value.strip().lower())


@tool
def remember_fact(
    kind: Literal["name", "likes", "preference", "bio"],
    value: str,
    runtime: ToolRuntime[AgentContext, Any],
    confidence: float = 0.8,
    source_text: str = "",
) -> dict[str, Any]:
    """Persist a durable user fact for future recall."""
    if runtime.store is None or runtime.context is None:
        return {"saved": False, "reason": "runtime store/context unavailable"}

    normalized_value = value.strip()
    if not normalized_value:
        return {"saved": False, "reason": "value cannot be empty"}

    namespace = _memory_namespace(runtime.context.user_id)
    existing = runtime.store.search(namespace, limit=100)
    requested_key = _dedupe_key(kind, normalized_value)
    for item in existing:
        if not isinstance(item.value, dict):
            continue
        existing_kind = item.value.get("kind")
        existing_value = item.value.get("value")
        if not isinstance(existing_kind, str) or not isinstance(existing_value, str):
            continue
        if _dedupe_key(existing_kind, existing_value) == requested_key:
            return {"saved": False, "duplicate": True, "fact": item.value}

    record = {
        "id": str(uuid.uuid4()),
        "user_id": runtime.context.user_id,
        "kind": kind,
        "value": normalized_value,
        "confidence": max(0.0, min(1.0, confidence)),
        "source_text": source_text.strip(),
    }
    runtime.store.put(namespace, record["id"], record)
    return {"saved": True, "duplicate": False, "fact": record}


@tool
def search_memories(
    runtime: ToolRuntime[AgentContext, Any],
    query: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search persisted user memories for the current user."""
    if runtime.store is None or runtime.context is None:
        return []

    bounded_limit = max(1, min(limit, 10))
    namespace = _memory_namespace(runtime.context.user_id)
    records = runtime.store.search(namespace, query=query or None, limit=bounded_limit)
    return [item.value for item in records if isinstance(item.value, dict)]


def build_agent_graph(
    *,
    model_name: str | None = None,
    checkpointer: InMemorySaver | None = None,
    store: InMemoryStore | None = None,
):
    saver = checkpointer or InMemorySaver()
    memory_store = store or InMemoryStore()
    model = _build_model(model_name=model_name)
    prompt = (
        "You are a persistence-focused assistant. "
        "Use search_memories before answering questions about the user's history or profile. "
        "When a user states durable personal facts (like name, likes, preferences, short bio), "
        "store them via remember_fact with the best-fitting kind. "
        "Avoid storing transient details."
    )
    graph = create_agent(
        model=model,
        tools=[search_memories, remember_fact],
        system_prompt=prompt,
        checkpointer=saver,
        store=memory_store,
        context_schema=AgentContext,
    )
    return graph, saver, memory_store


def build_graph(*, checkpointer: InMemorySaver | None = None, store: InMemoryStore | None = None):
    return build_agent_graph(checkpointer=checkpointer, store=store)


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
    payload = None if message is None else {"messages": [HumanMessage(content=message)]}
    state = graph.invoke(payload, config=config, context=AgentContext(user_id=user_id))
    messages = _extract_messages(state)
    return {
        "response": _latest_ai_text(messages),
        "messages": messages,
        "memory_events": _memory_events(messages),
    }


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


def _extract_messages(state: Any) -> list[BaseMessage]:
    if isinstance(state, dict):
        candidate = state.get("messages")
        if isinstance(candidate, list):
            return [m for m in candidate if isinstance(m, BaseMessage)]
    return []


def _latest_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                return msg.content
            return str(msg.content)
    return ""


def _memory_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "remember_fact":
            continue
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events
