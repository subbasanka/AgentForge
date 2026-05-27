from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"


class CostEntry(BaseModel):
    agent: AgentRole
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ResearchFinding(BaseModel):
    source: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    category: str
    summary: str
    key_insights: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewFeedback(BaseModel):
    section: str
    rating: int = Field(ge=1, le=5)
    feedback: str
    requires_revision: bool = False


class PipelineState(BaseModel):
    """Typed state schema shared across all agents in the graph."""

    # Core task
    task_id: str = ""
    query: str = ""
    company_or_topic: str = ""

    # Message history (LangGraph manages accumulation via add_messages)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Routing
    next_agent: AgentRole | str = AgentRole.SUPERVISOR
    iteration: int = 0
    max_iterations: int = 5

    # Research phase
    research_findings: Annotated[list[ResearchFinding], operator.add] = Field(default_factory=list)

    # Analysis phase
    analysis_results: Annotated[list[AnalysisResult], operator.add] = Field(default_factory=list)

    # Writing phase
    draft_report: str = ""

    # Review phase
    review_feedback: list[ReviewFeedback] = Field(default_factory=list)
    review_passed: bool = False

    # Final output
    final_report: str = ""

    # Human-in-the-loop
    pending_approval: bool = False
    approval_context: str = ""
    approved: bool | None = None

    # Cost tracking
    cost_ledger: Annotated[list[CostEntry], operator.add] = Field(default_factory=list)
    total_cost_usd: float = 0.0

    # Error tracking
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    agent_retry_counts: dict[str, int] = Field(default_factory=dict)

    # Completion
    completed: bool = False
    status: str = "pending"

    model_config = ConfigDict(arbitrary_types_allowed=True)
