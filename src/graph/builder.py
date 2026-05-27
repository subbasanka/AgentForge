from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import StateGraph, END

from src.agents.supervisor import supervisor_node
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.writer import writer_node
from src.agents.reviewer import reviewer_node
from src.graph.checkpoints import approval_gate_node
from src.middleware.retry import with_retry, RetryExhausted
from src.models.state import AgentRole, PipelineState
from src.observability.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def _route_after_supervisor(state: PipelineState) -> str:
    if state.pending_approval:
        return "approval_gate"
    next_val = state.next_agent
    if isinstance(next_val, AgentRole):
        next_val = next_val.value
    routes = {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "reviewer": "reviewer",
        "complete": "finalize",
    }
    return routes.get(next_val, "finalize")


def _route_after_approval(state: PipelineState) -> str:
    if state.pending_approval:
        return END
    return "supervisor"


async def _wrap_with_retry(node_fn, state: PipelineState, agent_name: str) -> dict:
    try:
        return await with_retry(node_fn, state, agent_name=agent_name)
    except RetryExhausted as exc:
        logger.error("Agent %s exhausted retries: %s", agent_name, exc)
        return {
            "errors": [str(exc)],
            "next_agent": AgentRole.SUPERVISOR,
        }


async def _researcher_with_retry(state: PipelineState) -> dict:
    return await _wrap_with_retry(researcher_node, state, "researcher")


async def _analyst_with_retry(state: PipelineState) -> dict:
    return await _wrap_with_retry(analyst_node, state, "analyst")


async def _writer_with_retry(state: PipelineState) -> dict:
    return await _wrap_with_retry(writer_node, state, "writer")


async def _reviewer_with_retry(state: PipelineState) -> dict:
    return await _wrap_with_retry(reviewer_node, state, "reviewer")


async def finalize_node(state: PipelineState) -> dict:
    total_cost = CostTracker.aggregate(state.cost_ledger)
    final = state.final_report or state.draft_report
    logger.info(
        "Pipeline complete: task=%s, cost=$%.6f, report_len=%d",
        state.task_id,
        total_cost,
        len(final),
    )
    return {
        "completed": True,
        "status": "completed",
        "final_report": final,
        "total_cost_usd": total_cost,
    }


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", _researcher_with_retry)
    graph.add_node("analyst", _analyst_with_retry)
    graph.add_node("writer", _writer_with_retry)
    graph.add_node("reviewer", _reviewer_with_retry)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "reviewer": "reviewer",
            "approval_gate": "approval_gate",
            "finalize": "finalize",
        },
    )

    for agent_node in ["researcher", "analyst", "writer", "reviewer"]:
        graph.add_edge(agent_node, "supervisor")

    graph.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    graph.add_edge("finalize", END)

    return graph


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


async def run_pipeline(
    query: str,
    company_or_topic: str = "",
    max_iterations: int = 5,
    task_id: str | None = None,
) -> PipelineState:
    graph = get_compiled_graph()

    initial_state = PipelineState(
        task_id=task_id or str(uuid.uuid4()),
        query=query,
        company_or_topic=company_or_topic or query,
        max_iterations=max_iterations,
    )

    final_state = await graph.ainvoke(initial_state)

    if isinstance(final_state, dict):
        return PipelineState(**final_state)
    return final_state
