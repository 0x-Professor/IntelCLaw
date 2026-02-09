"""
Tiny MCP stdio server for tests.

Run:
  python -m tests.fixtures.fake_mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

server = FastMCP("Fake MCP Server")

class EchoResult(BaseModel):
    echo: str


class AddResult(BaseModel):
    sum: int


@server.tool(
    name="Echo-Tool",
    description="Echo back input as structured content.",
    structured_output=True,
)
def echo(text: str) -> EchoResult:
    return EchoResult(echo=text)


@server.tool(
    description="Add two integers and return structured content.",
    structured_output=True,
)
def add(a: int, b: int) -> AddResult:
    return AddResult(sum=int(a) + int(b))


def main() -> None:
    # IMPORTANT: Do not print to stdout; stdio transport uses it for protocol messages.
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
