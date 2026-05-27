from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.agents.base import run_agent_llm
from src.models.state import AgentRole, AnalysisResult, PipelineState

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are an Analysis agent specializing in competitive intelligence synthesis.

Given research findings, produce structured analysis results organized by category.

Categories to analyze:
- Market Position: company's standing relative to competitors
- Competitive Landscape: key competitors and their strengths/weaknesses
- Strategic Direction: company's strategy and recent moves
- Financial Health: revenue, growth, profitability indicators
- Risks & Opportunities: threats and growth potential

For each category, provide:
- category: one of the categories above
- summary: concise analysis (2-3 sentences)
- key_insights: list of specific actionable insights
- confidence: 0.0 to 1.0 based on data quality

Respond with JSON:
{
    "analysis": [
        {
            "category": "Market Position",
            "summary": "...",
            "key_insights": ["insight1", "insight2"],
            "confidence": 0.85
        },
        ...
    ]
}"""


async def analyst_node(state: PipelineState) -> dict:
    findings_text = "\n\n".join(
        f"[{f.source}] (relevance: {f.relevance_score})\n{f.content}"
        for f in state.research_findings
    )

    user_message = (
        f"Company/Topic: {state.company_or_topic}\n"
        f"Research Query: {state.query}\n\n"
        f"Research Findings ({len(state.research_findings)} items):\n{findings_text}"
    )

    result = await run_agent_llm(
        role=AgentRole.ANALYST,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        user_message=user_message,
        state=state,
    )

    content = result["content"]
    analyses: list[AnalysisResult] = []

    try:
        parsed = json.loads(content)
        raw_analyses = parsed.get("analysis", [])
        for a in raw_analyses:
            analyses.append(AnalysisResult(
                category=a.get("category", "General"),
                summary=a.get("summary", ""),
                key_insights=a.get("key_insights", []),
                confidence=min(1.0, max(0.0, float(a.get("confidence", 0.7)))),
            ))
    except (json.JSONDecodeError, AttributeError, TypeError):
        analyses.append(AnalysisResult(
            category="General Analysis",
            summary=content[:500],
            key_insights=[],
            confidence=0.6,
        ))

    logger.info("Analyst produced %d analysis categories", len(analyses))

    return {
        "analysis_results": analyses,
        "cost_ledger": [result["cost_entry"]],
        "next_agent": AgentRole.SUPERVISOR,
        "messages": [AIMessage(content=f"Analyst: produced {len(analyses)} analysis categories")],
    }
