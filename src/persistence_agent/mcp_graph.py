from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from typing import Any, Mapping


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "http"
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    headers: Mapping[str, str] | None = None

    def to_client_entry(self) -> dict[str, Any]:
        if self.transport == "http":
            if not self.url:
                raise ValueError("MCPServerConfig requires 'url' when transport='http'.")
            entry: dict[str, Any] = {"transport": "http", "url": self.url}
            if self.headers:
                entry["headers"] = dict(self.headers)
            return entry

        if self.transport == "stdio":
            if not self.command:
                raise ValueError("MCPServerConfig requires 'command' when transport='stdio'.")
            return {"transport": "stdio", "command": self.command, "args": list(self.args)}

        raise ValueError(f"Unsupported MCP transport: {self.transport}")


def build_mcp_servers_from_env(env: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]] | None:
    source = env if env is not None else os.environ
    name = source.get("MCP_SERVER_NAME", "mcp")
    transport = source.get("MCP_TRANSPORT", "http").strip().lower()

    headers: dict[str, str] | None = None
    auth_header = source.get("MCP_AUTH_HEADER")
    auth_token = source.get("MCP_AUTH_TOKEN")
    if auth_header and auth_token:
        headers = {auth_header: auth_token}

    if transport == "http":
        url = source.get("MCP_SERVER_URL")
        if not url:
            return None
        config = MCPServerConfig(name=name, transport="http", url=url, headers=headers)
        return {name: config.to_client_entry()}

    if transport == "stdio":
        command = source.get("MCP_SERVER_COMMAND")
        if not command:
            return None
        args_raw = source.get("MCP_SERVER_ARGS", "")
        args = tuple(shlex.split(args_raw)) if args_raw else ()
        config = MCPServerConfig(name=name, transport="stdio", command=command, args=args)
        return {name: config.to_client_entry()}

    raise ValueError(f"Unsupported MCP transport from env: {transport}")


async def build_mcp_connected_graph(
    *,
    model: str | None = None,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
):
    servers = mcp_servers or build_mcp_servers_from_env()
    if not servers:
        raise ValueError("No MCP server configuration found. Set MCP_SERVER_URL or MCP_SERVER_COMMAND.")

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    model_name = model or f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
    graph = create_react_agent(model=model_name, tools=tools)
    return graph
