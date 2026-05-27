"""MCP server exposing file system and utility tools via HTTP JSON-RPC.

This implements the MCP tools/list and tools/call endpoints that agents
connect to for file I/O, data lookups, and external API access.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgentForge MCP Tool Server", version="1.0.0")

WORKSPACE_ROOT = os.getenv("MCP_WORKSPACE_ROOT", "/tmp/mcp_workspace")

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read contents of a file from the workspace",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within workspace"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within workspace"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files in a workspace directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path", "default": "."},
            },
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.md')"},
            },
            "required": ["pattern"],
        },
    },
]


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    id: int | str | None = None
    params: dict[str, Any] | None = None


def _resolve_path(relative: str) -> Path:
    base = Path(WORKSPACE_ROOT).resolve()
    target = (base / relative).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {relative}")
    return target


def _handle_read_file(args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_path(args["path"])
    if not path.exists():
        return {"content": [{"type": "text", "text": f"Error: file not found: {args['path']}"}], "isError": True}
    content = path.read_text(encoding="utf-8")
    return {"content": [{"type": "text", "text": content}]}


def _handle_write_file(args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return {"content": [{"type": "text", "text": f"Written {len(args['content'])} bytes to {args['path']}"}]}


def _handle_list_directory(args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_path(args.get("path", "."))
    if not path.is_dir():
        return {"content": [{"type": "text", "text": f"Error: not a directory: {args.get('path', '.')}"}], "isError": True}
    entries = sorted(
        f"{'[dir]' if p.is_dir() else '[file]'} {p.name}"
        for p in path.iterdir()
    )
    return {"content": [{"type": "text", "text": "\n".join(entries) if entries else "(empty directory)"}]}


def _handle_search_files(args: dict[str, Any]) -> dict[str, Any]:
    base = Path(WORKSPACE_ROOT).resolve()
    matches = sorted(str(p.relative_to(base)) for p in base.rglob(args["pattern"]))
    return {"content": [{"type": "text", "text": "\n".join(matches) if matches else "(no matches)"}]}


_TOOL_HANDLERS = {
    "read_file": _handle_read_file,
    "write_file": _handle_write_file,
    "list_directory": _handle_list_directory,
    "search_files": _handle_search_files,
}


@app.post("/mcp/v1/tools/list")
async def tools_list(request: JsonRpcRequest) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "result": {"tools": TOOL_DEFINITIONS},
    }


@app.post("/mcp/v1/tools/call")
async def tools_call(request: JsonRpcRequest) -> dict[str, Any]:
    params = request.params or {}
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }

    try:
        result = handler(arguments)
        return {"jsonrpc": "2.0", "id": request.id, "result": result}
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc)
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32000, "message": str(exc)},
        }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "mcp-tool-server"}


if __name__ == "__main__":
    import uvicorn

    Path(WORKSPACE_ROOT).mkdir(parents=True, exist_ok=True)
    uvicorn.run("mcp_server.server:app", host="0.0.0.0", port=8001, reload=True)
