from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import AuthenticationError


pytestmark = pytest.mark.integration


def test_openai_chat_model_connectivity() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    try:
        response = llm.invoke([HumanMessage(content="Reply with the exact token: OK")])
    except AuthenticationError:
        raise AssertionError("OpenAI authentication failed. Check OPENAI_API_KEY in .env.") from None
    content = response.content if isinstance(response.content, str) else str(response.content)

    assert isinstance(content, str)
    assert content.strip()
