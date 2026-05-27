from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
import src.observability.langfuse_client as _lf_mod


@pytest.fixture(autouse=True)
def _reset_langfuse_singleton():
    _lf_mod._langfuse_instance = None
    yield
    _lf_mod._langfuse_instance = None


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses with proper structure."""

    def _make(content: str, input_tokens: int = 100, output_tokens: int = 200):
        response = MagicMock()
        response.content = content
        response.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        response.tool_calls = []
        return response

    return _make


@pytest.fixture
def mock_supervisor_responses(mock_llm_response):
    """Sequence of supervisor routing decisions for a full pipeline run."""
    return [
        mock_llm_response(json.dumps({"next_agent": "researcher"})),
        mock_llm_response(json.dumps({"next_agent": "analyst"})),
        mock_llm_response(json.dumps({"next_agent": "writer"})),
        mock_llm_response(json.dumps({"next_agent": "reviewer"})),
        mock_llm_response(json.dumps({"next_agent": "complete"})),
    ]


@pytest.fixture
def mock_researcher_response(mock_llm_response):
    return mock_llm_response(json.dumps({
        "findings": [
            {
                "source": "web_search",
                "content": "Acme Corp holds 35% market share in the widget industry.",
                "relevance_score": 0.95,
            },
            {
                "source": "industry_report",
                "content": "The widget market is projected to grow 12% annually through 2027.",
                "relevance_score": 0.88,
            },
        ],
        "summary": "Gathered key findings about Acme Corp market position and industry trends.",
    }))


@pytest.fixture
def mock_analyst_response(mock_llm_response):
    return mock_llm_response(json.dumps({
        "analysis": [
            {
                "category": "Market Position",
                "summary": "Acme Corp is the market leader with 35% share.",
                "key_insights": ["Dominant market position", "Growing sector"],
                "confidence": 0.9,
            },
            {
                "category": "Competitive Landscape",
                "summary": "Two main competitors with 20% and 15% share respectively.",
                "key_insights": ["Fragmented competition", "Barriers to entry exist"],
                "confidence": 0.82,
            },
        ],
    }))


@pytest.fixture
def mock_writer_response(mock_llm_response):
    return mock_llm_response(
        "# Competitive Analysis: Acme Corp\n\n"
        "## Executive Summary\n"
        "Acme Corp maintains a dominant market position with 35% share.\n\n"
        "## Market Position\nLeading player in the widget industry.\n\n"
        "## Competitive Landscape\nFragmented competition with two main rivals.\n\n"
        "## Conclusions\nStrong position with growth opportunities."
    )


@pytest.fixture
def mock_reviewer_response(mock_llm_response):
    return mock_llm_response(json.dumps({
        "review": [
            {
                "section": "Executive Summary",
                "rating": 4,
                "feedback": "Clear and concise summary.",
                "requires_revision": False,
            },
            {
                "section": "Market Position",
                "rating": 4,
                "feedback": "Well supported by data.",
                "requires_revision": False,
            },
        ],
        "overall_pass": True,
    }))


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
