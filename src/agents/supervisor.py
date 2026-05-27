from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.agents.base import run_agent_llm
from src.models.state import AgentRole, PipelineState

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are a Supervisor agent orchestrating a competitive research pipeline.
Your job is to route work to specialized agents and decide when the pipeline is complete.

Available agents:
- researcher: Gathers information from web searches and files. Use first to collect raw data.
- analyst: Processes research findings into structured insights with confidence scores.
- writer: Produces a polished competitive analysis report from the analysis.
- reviewer: Reviews the draft report for quality, accuracy, and completeness.

Current pipeline rules:
1. Always start with the researcher to gather data.
2. After research, route to the analyst for structured analysis.
3. After analysis, route to the writer to produce a draft report.
4. After writing, route to the reviewer for quality check.
5. If the reviewer flags issues, route back to the writer for revision.
6. If the review passes, mark the pipeline as complete.

You must respond with valid JSON containing exactly one key: "next_agent"
The value must be one of: "researcher", "analyst", "writer", "reviewer", "complete"

Respond ONLY with the JSON object, nothing else."""


async def supervisor_node(state: PipelineState) -> dict:
    has_research = len(state.research_findings) > 0
    has_analysis = len(state.analysis_results) > 0
    has_draft = bool(state.draft_report)
    has_review = len(state.review_feedback) > 0

    context_parts = [
        f"Task: {state.query}",
        f"Topic: {state.company_or_topic}",
        f"Iteration: {state.iteration}/{state.max_iterations}",
        f"Has research: {has_research} ({len(state.research_findings)} findings)",
        f"Has analysis: {has_analysis} ({len(state.analysis_results)} results)",
        f"Has draft: {has_draft}",
        f"Has review: {has_review}",
        f"Review passed: {state.review_passed}",
        f"Errors: {state.errors[-3:] if state.errors else 'none'}",
    ]
    user_message = "\n".join(context_parts)

    if state.iteration >= state.max_iterations:
        logger.info("Max iterations reached, completing pipeline")
        return {
            "next_agent": "complete",
            "messages": [AIMessage(content="Max iterations reached. Finalizing pipeline.")],
            "iteration": state.iteration + 1,
        }

    result = await run_agent_llm(
        role=AgentRole.SUPERVISOR,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        user_message=user_message,
        state=state,
    )

    content = result["content"].strip()
    try:
        parsed = json.loads(content)
        next_agent = parsed.get("next_agent", "researcher")
    except (json.JSONDecodeError, AttributeError):
        if state.review_passed:
            next_agent = "complete"
        elif has_review and not state.review_passed:
            next_agent = "writer"
        elif has_draft:
            next_agent = "reviewer"
        elif has_analysis:
            next_agent = "writer"
        elif has_research:
            next_agent = "analyst"
        else:
            next_agent = "researcher"

    logger.info("Supervisor routing to: %s (iteration %d)", next_agent, state.iteration + 1)

    return {
        "next_agent": next_agent,
        "cost_ledger": [result["cost_entry"]],
        "messages": [AIMessage(content=f"Supervisor: routing to {next_agent}")],
        "iteration": state.iteration + 1,
    }
