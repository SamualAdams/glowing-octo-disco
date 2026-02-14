from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol


class LLMResponder(Protocol):
    def __call__(self, *, user_message: str, memory_summary: str) -> str: ...


@dataclass(frozen=True)
class OpenAIModelConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 20.0


def _default_fallback(user_message: str, memory_summary: str) -> str:
    return f"You said: {user_message}\nMemory: {memory_summary}"


def build_openai_responder(config: OpenAIModelConfig) -> LLMResponder:
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, timeout=config.timeout_seconds)

    def respond(*, user_message: str, memory_summary: str) -> str:
        completion = client.chat.completions.create(
            model=config.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise assistant for a persistence demo. "
                        "Respond in 1-2 short lines, grounded in the provided memory summary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User message: {user_message}\nMemory summary: {memory_summary}",
                },
            ],
        )
        content = completion.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        return _default_fallback(user_message, memory_summary)

    return respond


def build_openai_responder_from_env() -> LLMResponder | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return build_openai_responder(OpenAIModelConfig(api_key=api_key, model=model))
