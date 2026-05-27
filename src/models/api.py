from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.state import CostEntry, ReviewFeedback


class RunRequest(BaseModel):
    query: str = Field(description="Research query or company name to analyze")
    company_or_topic: str = Field(default="", description="Target company or topic")
    max_iterations: int = Field(default=5, ge=1, le=20)


class RunStatus(BaseModel):
    task_id: str
    status: str
    current_agent: str = ""
    iteration: int = 0
    pending_approval: bool = False
    approval_context: str = ""
    cost_so_far: float = 0.0
    errors: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    task_id: str
    status: str
    final_report: str = ""
    cost_ledger: list[CostEntry] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    review_feedback: list[ReviewFeedback] = Field(default_factory=list)
    iterations: int = 0
    errors: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    task_id: str
    approved: bool
    reason: str = ""


class ApprovalResponse(BaseModel):
    task_id: str
    status: str
    message: str
