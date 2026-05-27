from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.agents.base import run_agent_llm
from src.models.state import AgentRole, PipelineState

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a Writer agent specializing in producing polished competitive analysis reports.

Given analysis results and optional reviewer feedback, produce a comprehensive report.

Report structure:
1. Executive Summary (2-3 paragraphs)
2. Market Position Analysis
3. Competitive Landscape
4. Strategic Analysis
5. Financial Overview
6. Risks & Opportunities
7. Conclusions & Recommendations

Guidelines:
- Write in professional business language
- Support claims with data from the analysis
- Include confidence levels where appropriate
- If reviewer feedback is provided, address all flagged issues
- Aim for 800-1500 words

Respond with the complete report text. Do NOT wrap in JSON."""

REVISION_ADDENDUM = """

IMPORTANT: The previous draft was reviewed and received the following feedback.
Address ALL issues raised by the reviewer:

{feedback}

Produce a revised version of the report that incorporates this feedback."""


async def writer_node(state: PipelineState) -> dict:
    analysis_text = "\n\n".join(
        f"## {a.category} (confidence: {a.confidence})\n"
        f"{a.summary}\n"
        f"Key insights: {', '.join(a.key_insights)}"
        for a in state.analysis_results
    )

    system_prompt = WRITER_SYSTEM_PROMPT

    if state.review_feedback and not state.review_passed:
        feedback_text = "\n".join(
            f"- [{fb.section}] Rating: {fb.rating}/5 — {fb.feedback}"
            for fb in state.review_feedback
        )
        system_prompt += REVISION_ADDENDUM.format(feedback=feedback_text)

    user_message = (
        f"Company/Topic: {state.company_or_topic}\n"
        f"Research Query: {state.query}\n\n"
        f"Analysis Results:\n{analysis_text}"
    )

    if state.draft_report:
        user_message += f"\n\nPrevious Draft:\n{state.draft_report[:3000]}"

    result = await run_agent_llm(
        role=AgentRole.WRITER,
        system_prompt=system_prompt,
        user_message=user_message,
        state=state,
        bind_tools=True,
    )

    draft = result["content"]

    # If the draft text is empty but the model chose to write a file,
    # extract the draft report directly from the tool call arguments!
    if not draft:
        for tc in result.get("tool_calls", []):
            if tc.get("tool_name") == "file_write_tool":
                draft = tc.get("arguments", {}).get("content", "")
                break

    approval_needed = any(
        tc.get("requires_approval") for tc in result.get("tool_calls", [])
    )

    update: dict = {
        "draft_report": draft,
        "cost_ledger": [result["cost_entry"]],
        "next_agent": AgentRole.SUPERVISOR,
        "messages": [AIMessage(content=f"Writer: produced draft ({len(draft)} chars)")],
    }

    if approval_needed:
        update["pending_approval"] = True
        update["approval_context"] = (
            "Writer agent wants to save the report to disk. "
            "Approve to allow file write, or reject to skip."
        )

    logger.info("Writer produced draft: %d chars, approval_needed=%s", len(draft), approval_needed)
    return update
