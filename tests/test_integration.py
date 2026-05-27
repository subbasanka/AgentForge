"""End-to-end integration tests for the multi-agent pipeline.

Runs the full LangGraph against mock LLM responses and asserts on:
- Final state correctness (report generated, review passed)
- Cost tracked across all agents
- All agent nodes executed in correct order
- Error recovery via retry logic
- Token budget enforcement
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.builder import build_graph
from src.middleware.retry import RetryExhausted, with_retry
from src.middleware.token_budget import TokenBudgetExceeded, TokenBudgetGuard
from src.models.state import AgentRole, CostEntry, PipelineState
from src.observability.cost_tracker import CostTracker


def _make_mock_response(content: str, input_tokens: int = 100, output_tokens: int = 200):
    resp = MagicMock()
    resp.content = content
    resp.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    resp.tool_calls = []
    return resp


def _build_llm_side_effects():
    """Build a sequence of LLM responses that drives the pipeline through all phases."""
    return [
        # Supervisor → researcher
        _make_mock_response(json.dumps({"next_agent": "researcher"})),
        # Researcher execution
        _make_mock_response(json.dumps({
            "findings": [
                {"source": "web", "content": "Acme Corp leads with 35% market share.", "relevance_score": 0.95},
                {"source": "report", "content": "Market growing 12% annually.", "relevance_score": 0.88},
            ],
            "summary": "Key findings gathered.",
        })),
        # Supervisor → analyst
        _make_mock_response(json.dumps({"next_agent": "analyst"})),
        # Analyst execution
        _make_mock_response(json.dumps({
            "analysis": [
                {"category": "Market Position", "summary": "Market leader.", "key_insights": ["35% share"], "confidence": 0.9},
                {"category": "Competition", "summary": "Fragmented.", "key_insights": ["Two rivals"], "confidence": 0.8},
            ],
        })),
        # Supervisor → writer
        _make_mock_response(json.dumps({"next_agent": "writer"})),
        # Writer execution
        _make_mock_response("# Report\n\n## Summary\nAcme leads the market.\n\n## Analysis\nStrong position."),
        # Supervisor → reviewer
        _make_mock_response(json.dumps({"next_agent": "reviewer"})),
        # Reviewer execution (passes)
        _make_mock_response(json.dumps({
            "review": [{"section": "Summary", "rating": 4, "feedback": "Good.", "requires_revision": False}],
            "overall_pass": True,
        })),
        # Supervisor → complete
        _make_mock_response(json.dumps({"next_agent": "complete"})),
    ]


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end():
    """Run the complete pipeline with mocked LLM and verify final state."""
    side_effects = _build_llm_side_effects()
    mock_llm = AsyncMock(side_effect=side_effects)

    with patch("src.agents.base.get_llm") as mock_get_llm:
        llm_instance = MagicMock()
        llm_instance.ainvoke = mock_llm
        llm_instance.bind_tools = MagicMock(return_value=llm_instance)
        mock_get_llm.return_value = llm_instance

        graph = build_graph().compile()

        initial_state = PipelineState(
            task_id="test-001",
            query="Analyze Acme Corp competitive position",
            company_or_topic="Acme Corp",
            max_iterations=10,
        )

        result = await graph.ainvoke(initial_state)

    assert result["completed"] is True
    assert result["status"] == "completed"
    assert len(result["final_report"]) > 0
    assert result["review_passed"] is True
    assert len(result["research_findings"]) > 0
    assert len(result["analysis_results"]) > 0
    assert len(result["cost_ledger"]) > 0
    assert result["total_cost_usd"] > 0

    cost_agents = {entry.agent if isinstance(entry, CostEntry) else entry["agent"] for entry in result["cost_ledger"]}
    for role in [AgentRole.SUPERVISOR, AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.WRITER, AgentRole.REVIEWER]:
        assert role in cost_agents or role.value in cost_agents, f"Missing cost entry for {role}"


@pytest.mark.asyncio
async def test_pipeline_respects_max_iterations():
    """Verify the pipeline stops at max_iterations even if not converged."""
    supervisor_response = _make_mock_response(json.dumps({"next_agent": "researcher"}))
    researcher_response = _make_mock_response(json.dumps({
        "findings": [{"source": "web", "content": "Data.", "relevance_score": 0.7}],
        "summary": "Found data.",
    }))

    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return supervisor_response
        return researcher_response

    mock_llm = AsyncMock(side_effect=_side_effect)

    with patch("src.agents.base.get_llm") as mock_get_llm:
        llm_instance = MagicMock()
        llm_instance.ainvoke = mock_llm
        llm_instance.bind_tools = MagicMock(return_value=llm_instance)
        mock_get_llm.return_value = llm_instance

        graph = build_graph().compile()
        initial_state = PipelineState(
            task_id="test-max-iter",
            query="Test iteration limit",
            company_or_topic="Test",
            max_iterations=3,
        )

        result = await graph.ainvoke(initial_state)

    assert result["completed"] is True
    assert result["iteration"] <= 4  # max_iterations + 1 for the finalize step


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff():
    """Verify retry logic with exponential backoff."""
    call_count = 0

    async def _flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Transient failure #{call_count}")
        return "success"

    result = await with_retry(
        _flaky_fn,
        agent_name="test_agent",
        max_retries=3,
        base_delay=0.01,
    )
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """Verify RetryExhausted is raised when all retries fail."""
    async def _always_fails():
        raise ValueError("permanent failure")

    with pytest.raises(RetryExhausted) as exc_info:
        await with_retry(
            _always_fails,
            agent_name="failing_agent",
            max_retries=2,
            base_delay=0.01,
        )
    assert exc_info.value.agent == "failing_agent"
    assert exc_info.value.attempts == 2


def test_token_budget_truncation():
    """Verify token budget guard truncates text correctly."""
    guard = TokenBudgetGuard()
    long_text = "word " * 5000  # ~5000 tokens
    truncated = guard.truncate_to_budget(long_text, budget=1000, reserve=100)
    token_count = guard.count_tokens(truncated)
    assert token_count <= 900


def test_token_budget_exceeded():
    """Verify TokenBudgetExceeded is raised when budget is blown."""
    guard = TokenBudgetGuard()
    long_text = "word " * 5000
    with pytest.raises(TokenBudgetExceeded):
        guard.check_budget("test_agent", long_text, budget=100)


def test_cost_tracker():
    """Verify cost computation and aggregation."""
    tracker = CostTracker(input_cost_per_m=2.50, output_cost_per_m=10.00)

    entry = tracker.record(AgentRole.RESEARCHER, input_tokens=1000, output_tokens=500)
    assert entry.input_tokens == 1000
    assert entry.output_tokens == 500
    expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
    assert abs(entry.cost_usd - expected_cost) < 1e-8

    entries = [
        tracker.record(AgentRole.RESEARCHER, 1000, 500),
        tracker.record(AgentRole.ANALYST, 2000, 1000),
    ]
    total = CostTracker.aggregate(entries)
    assert total > 0


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Verify the health check endpoint responds correctly."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_run_endpoint_starts_pipeline(async_client):
    """Verify the /run endpoint accepts requests and returns a task ID."""
    side_effects = _build_llm_side_effects()
    mock_llm = AsyncMock(side_effect=side_effects)

    with patch("src.agents.base.get_llm") as mock_get_llm:
        llm_instance = MagicMock()
        llm_instance.ainvoke = mock_llm
        llm_instance.bind_tools = MagicMock(return_value=llm_instance)
        mock_get_llm.return_value = llm_instance

        response = await async_client.post(
            "/api/v1/run",
            json={
                "query": "Analyze Acme Corp",
                "company_or_topic": "Acme Corp",
                "max_iterations": 5,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "started"
