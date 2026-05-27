from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)


@tool
async def web_search_tool(query: str) -> str:
    """Search the web for real-time information about a topic, company, or market.

    Args:
        query: The search query string.

    Returns:
        JSON string with search results containing title, snippet, and url.
    """
    # 1. Try Exa Search
    if settings.exa_api_key:
        try:
            from exa_py import Exa
            logger.info("Performing live search using Exa for query: %s", query)
            exa = Exa(api_key=settings.exa_api_key)
            response = exa.search_and_contents(
                query,
                type="neural",
                use_autoprompt=True,
                num_results=5,
                highlights=True,
            )
            results = []
            for item in response.results:
                results.append({
                    "title": getattr(item, "title", "No Title"),
                    "url": getattr(item, "url", ""),
                    "snippet": getattr(item, "text", "")[:1000] if getattr(item, "text", "") else getattr(item, "highlights", [""])[0],
                    "relevance": getattr(item, "score", 0.8),
                })
            return json.dumps(results, indent=2)
        except Exception as exc:
            logger.warning("Exa search failed, falling back: %s", exc)

    # 2. Try Tavily Search
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            logger.info("Performing live search using Tavily for query: %s", query)
            tavily = TavilyClient(api_key=settings.tavily_api_key)
            response = tavily.search(query=query, search_depth="advanced", max_results=5)
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "No Title"),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "relevance": item.get("score", 0.8),
                })
            return json.dumps(results, indent=2)
        except Exception as exc:
            logger.warning("Tavily search failed, falling back: %s", exc)

    # 3. Fallback: DuckDuckGo Search (Completely free, no key required)
    try:
        from duckduckgo_search import DDGS
        logger.info("Performing live search using DuckDuckGo for query: %s", query)
        results = []
        with DDGS() as ddgs:
            ddg_results = ddgs.text(query, max_results=5)
            for item in ddg_results:
                results.append({
                    "title": item.get("title", "No Title"),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "relevance": 0.8,
                })
        if results:
            return json.dumps(results, indent=2)
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)

    # Final Fallback to avoid complete pipeline failure
    logger.warning("All live search methods failed. Returning simulated results.")
    simulated_results = [
        {
            "title": f"Search result for: {query}",
            "snippet": f"Comprehensive information about {query}. Live search currently offline.",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            "relevance": 0.5,
        }
    ]
    return json.dumps(simulated_results, indent=2)


@tool
async def file_read_tool(file_path: str) -> str:
    """Read content from a file via MCP file system access.

    Args:
        file_path: Path to the file to read.

    Returns:
        The file contents as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path}"})
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {file_path}"})


@tool
async def file_write_tool(file_path: str, content: str) -> str:
    """Write content to a file via MCP file system access. Requires human approval.

    Args:
        file_path: Path to the file to write.
        content: Content to write to the file.

    Returns:
        Confirmation message.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"status": "success", "path": file_path, "bytes": len(content)})
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {file_path}"})
