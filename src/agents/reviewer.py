from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.agents.base import run_agent_llm
from src.models.state import AgentRole, PipelineState, ReviewFeedback

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a Quality Review agent for competitive analysis reports.

Evaluate the draft report on these dimensions:
1. Executive Summary — Does it capture the key points?
2. Analysis Depth — Are claims well-supported?
3. Completeness — Are all required sections present?
4. Accuracy — Are there factual inconsistencies?
5. Actionability — Are recommendations specific and useful?

For each section, provide:
- section: name of the section being reviewed
- rating: 1-5 (1=poor, 5=excellent)
- feedback: specific improvement suggestions
- requires_revision: true if the section needs rework

The report passes review if ALL sections score >= 3 and no section requires_revision.

Respond with JSON:
{
    "review": [
        {
            "section": "Executive Summary",
            "rating": 4,
            "feedback": "Clear and concise, but could mention key competitor names.",
            "requires_revision": false
        },
        ...
    ],
    "overall_pass": true
}"""


async def reviewer_node(state: PipelineState) -> dict:
    user_message = (
        f"Company/Topic: {state.company_or_topic}\n"
        f"Research Query: {state.query}\n\n"
        f"Draft Report:\n{state.draft_report}\n\n"
        f"Number of research findings used: {len(state.research_findings)}\n"
        f"Number of analysis categories: {len(state.analysis_results)}"
    )

    result = await run_agent_llm(
        role=AgentRole.REVIEWER,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_message=user_message,
        state=state,
    )

    content = result["content"]
    feedback_items: list[ReviewFeedback] = []
    passed = False

    try:
        parsed = json.loads(content)
        raw_reviews = parsed.get("review", [])
        for r in raw_reviews:
            feedback_items.append(ReviewFeedback(
                section=r.get("section", "General"),
                rating=max(1, min(5, int(r.get("rating", 3)))),
                feedback=r.get("feedback", ""),
                requires_revision=r.get("requires_revision", False),
            ))
        passed = parsed.get("overall_pass", False)
        if not isinstance(passed, bool):
            passed = all(f.rating >= 3 and not f.requires_revision for f in feedback_items)
    except (json.JSONDecodeError, AttributeError, TypeError):
        feedback_items.append(ReviewFeedback(
            section="General",
            rating=3,
            feedback=content[:500],
            requires_revision=False,
        ))
        passed = True

    logger.info("Reviewer: pass=%s, %d feedback items", passed, len(feedback_items))

    return {
        "review_feedback": feedback_items,
        "review_passed": passed,
        "final_report": state.draft_report if passed else "",
        "cost_ledger": [result["cost_entry"]],
        "next_agent": AgentRole.SUPERVISOR,
        "messages": [AIMessage(
            content=f"Reviewer: {'PASSED' if passed else 'REVISION NEEDED'} ({len(feedback_items)} items)"
        )],
    }
