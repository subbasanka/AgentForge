from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.agents.base import run_agent_llm
from src.models.state import AgentRole, PipelineState, ResearchFinding

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a Research agent specializing in competitive intelligence gathering.

Your task is to research the given company or topic and produce structured findings.

For each finding, provide:
- source: where the information came from
- content: the key information discovered
- relevance_score: 0.0 to 1.0 indicating relevance to the research query

You have access to web_search_tool and file_read_tool.

Respond with a JSON object containing:
{
    "findings": [
        {"source": "...", "content": "...", "relevance_score": 0.95},
        ...
    ],
    "summary": "Brief summary of what was found"
}

Research thoroughly — cover market position, competitors, recent developments, financials, and strategic direction."""


async def researcher_node(state: PipelineState) -> dict:
    user_message = (
        f"Research query: {state.query}\n"
        f"Company/Topic: {state.company_or_topic}\n"
        f"Existing findings: {len(state.research_findings)}"
    )

    result = await run_agent_llm(
        role=AgentRole.RESEARCHER,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        user_message=user_message,
        state=state,
        bind_tools=True,
    )

    content = result["content"]
    findings: list[ResearchFinding] = []

    try:
        parsed = json.loads(content)
        raw_findings = parsed.get("findings", [])
        for f in raw_findings:
            findings.append(ResearchFinding(
                source=f.get("source", "LLM analysis"),
                content=f.get("content", ""),
                relevance_score=min(1.0, max(0.0, float(f.get("relevance_score", 0.7)))),
            ))
    except (json.JSONDecodeError, AttributeError, TypeError):
        findings.append(ResearchFinding(
            source="LLM direct response",
            content=content,
            relevance_score=0.75,
        ))

    for tc in result.get("tool_calls", []):
        if tc.get("success") and tc.get("result"):
            findings.append(ResearchFinding(
                source=f"tool:{tc['tool_name']}",
                content=tc["result"][:2000],
                relevance_score=0.85,
            ))

    logger.info("Researcher produced %d findings", len(findings))

    return {
        "research_findings": findings,
        "cost_ledger": [result["cost_entry"]],
        "next_agent": AgentRole.SUPERVISOR,
        "messages": [AIMessage(content=f"Researcher: gathered {len(findings)} findings")],
    }
