from __future__ import annotations

import pytest

from persistence_agent.mcp_graph import MCPServerConfig, build_mcp_servers_from_env


def test_http_server_config_from_env() -> None:
    env = {
        "MCP_SERVER_NAME": "weather",
        "MCP_TRANSPORT": "http",
        "MCP_SERVER_URL": "http://localhost:8000/mcp",
        "MCP_AUTH_HEADER": "X-Api-Key",
        "MCP_AUTH_TOKEN": "secret",
    }

    config = build_mcp_servers_from_env(env)
    assert config == {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {"X-Api-Key": "secret"},
        }
    }


def test_stdio_server_config_from_env() -> None:
    env = {
        "MCP_SERVER_NAME": "math",
        "MCP_TRANSPORT": "stdio",
        "MCP_SERVER_COMMAND": "python",
        "MCP_SERVER_ARGS": "server.py --mode local",
    }

    config = build_mcp_servers_from_env(env)
    assert config == {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py", "--mode", "local"],
        }
    }


def test_missing_http_url_returns_none() -> None:
    env = {"MCP_TRANSPORT": "http"}
    assert build_mcp_servers_from_env(env) is None


def test_unsupported_transport_raises() -> None:
    with pytest.raises(ValueError):
        build_mcp_servers_from_env({"MCP_TRANSPORT": "sse"})


def test_mcp_server_config_validation() -> None:
    with pytest.raises(ValueError):
        MCPServerConfig(name="x", transport="http", url=None).to_client_entry()
