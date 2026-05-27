from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class MCPToolClient:
    """Client that dispatches tool calls to an MCP server over HTTP.

    The MCP server exposes tools (file system, database, APIs) via a standard
    JSON-RPC interface. This client wraps that interface for use by agents.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.mcp_server_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/mcp/v1/tools/list",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", {}).get("tools", [])
        except httpx.HTTPError as exc:
            logger.warning("MCP tools/list failed: %s", exc)
            return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/mcp/v1/tools/call",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": 1,
                    "params": {"name": tool_name, "arguments": arguments or {}},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return {"error": data["error"].get("message", "Unknown MCP error")}
            return data.get("result", {})
        except httpx.HTTPError as exc:
            logger.error("MCP tool call '%s' failed: %s", tool_name, exc)
            return {"error": str(exc)}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
