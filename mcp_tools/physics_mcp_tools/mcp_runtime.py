"""Shared runtime helpers for Physics Assistant MCP servers."""

from typing import Any


def run_fastmcp_server(mcp: Any, host: str, port: int, transport: str) -> None:
    """Run a FastMCP server with the transport spellings used by this project."""
    normalized_transport = transport.replace("_", "-")

    if normalized_transport == "stdio":
        mcp.run(transport="stdio")
        return

    if normalized_transport == "sse":
        app_provider = getattr(mcp, "sse_http_app", None) or getattr(
            mcp, "sse_app", None
        )
    elif normalized_transport == "streamable-http":
        app_provider = getattr(mcp, "streamable_http_app", None)
    else:
        raise ValueError(
            "Unsupported MCP transport. Use stdio, sse, or streamable_http."
        )

    if app_provider is None:
        raise RuntimeError(f"FastMCP does not expose an app for {transport} transport")

    app = app_provider() if callable(app_provider) else app_provider

    import uvicorn

    uvicorn.run(app, host=host, port=port)
