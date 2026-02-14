from __future__ import annotations

import asyncio

from persistence_agent.mcp_graph import build_mcp_connected_graph


async def main() -> None:
    graph = await build_mcp_connected_graph()
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What tools do you have available?"}]}
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
