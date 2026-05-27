from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from src.graph.builder import get_compiled_graph
from src.models.api import (
    ApprovalRequest,
    ApprovalResponse,
    RunRequest,
    RunResponse,
    RunStatus,
)
from src.models.state import PipelineState
from src.observability.cost_tracker import CostTracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["agentforge"])

_active_runs: dict[str, dict[str, Any]] = {}


@router.post("/run", response_model=RunResponse)
async def start_run(request: RunRequest) -> RunResponse:
    task_id = str(uuid.uuid4())
    graph = get_compiled_graph()

    initial_state = PipelineState(
        task_id=task_id,
        query=request.query,
        company_or_topic=request.company_or_topic or request.query,
        max_iterations=request.max_iterations,
    )

    _active_runs[task_id] = {"state": initial_state, "task": None, "paused": False}

    async def _execute() -> None:
        try:
            result = await graph.ainvoke(initial_state)
            if isinstance(result, dict):
                _active_runs[task_id]["state"] = PipelineState(**result)
            else:
                _active_runs[task_id]["state"] = result
        except Exception as exc:
            logger.error("Pipeline %s failed: %s", task_id, exc)
            state = _active_runs[task_id]["state"]
            state.status = "failed"
            state.errors.append(str(exc))

    task = asyncio.create_task(_execute())
    _active_runs[task_id]["task"] = task

    return RunResponse(
        task_id=task_id,
        status="started",
    )


@router.get("/run/{task_id}", response_model=RunStatus)
async def get_run_status(task_id: str) -> RunStatus:
    if task_id not in _active_runs:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    state: PipelineState = _active_runs[task_id]["state"]
    task: asyncio.Task | None = _active_runs[task_id].get("task")

    status = state.status
    if state.pending_approval:
        status = "awaiting_approval"
    elif task and task.done():
        status = "completed" if state.completed else "failed"
    elif task and not task.done():
        status = "running"

    return RunStatus(
        task_id=task_id,
        status=status,
        current_agent=state.next_agent if isinstance(state.next_agent, str) else state.next_agent.value,
        iteration=state.iteration,
        pending_approval=state.pending_approval,
        approval_context=state.approval_context,
        cost_so_far=CostTracker.aggregate(state.cost_ledger),
        errors=state.errors,
    )


@router.get("/run/{task_id}/result", response_model=RunResponse)
async def get_run_result(task_id: str) -> RunResponse:
    if task_id not in _active_runs:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    state: PipelineState = _active_runs[task_id]["state"]

    return RunResponse(
        task_id=task_id,
        status=state.status,
        final_report=state.final_report,
        cost_ledger=state.cost_ledger,
        total_cost_usd=state.total_cost_usd,
        review_feedback=state.review_feedback,
        iterations=state.iteration,
        errors=state.errors,
    )


@router.post("/run/{task_id}/approve", response_model=ApprovalResponse)
async def approve_action(task_id: str, request: ApprovalRequest) -> ApprovalResponse:
    if task_id not in _active_runs:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    state: PipelineState = _active_runs[task_id]["state"]

    if not state.pending_approval:
        raise HTTPException(status_code=400, detail="No pending approval for this task")

    state.approved = request.approved
    state.pending_approval = False

    action = "approved" if request.approved else "denied"
    logger.info("Task %s: human %s action. Re-invoking graph to resume.", task_id, action)

    # Start a new background task to resume the graph with the updated state
    graph = get_compiled_graph()

    async def _execute() -> None:
        try:
            result = await graph.ainvoke(state)
            if isinstance(result, dict):
                _active_runs[task_id]["state"] = PipelineState(**result)
            else:
                _active_runs[task_id]["state"] = result
        except Exception as exc:
            logger.error("Pipeline %s failed on resume: %s", task_id, exc)
            state.status = "failed"
            state.errors.append(str(exc))

    task = asyncio.create_task(_execute())
    _active_runs[task_id]["task"] = task

    return ApprovalResponse(
        task_id=task_id,
        status=f"action_{action}",
        message=f"Action {action}. Pipeline will resume.",
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "agentforge"}
