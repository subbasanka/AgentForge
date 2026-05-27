from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.state import AgentRole


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result: str = ""
    success: bool = True
    duration_ms: float = 0.0


class CostRecord(BaseModel):
    agent: AgentRole
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


class AgentInput(BaseModel):
    task_description: str
    context: str = ""
    tools_available: list[str] = Field(default_factory=list)
    token_budget: int = 8000


class AgentOutput(BaseModel):
    agent: AgentRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    cost: CostRecord
    success: bool = True
    error: str | None = None
    requires_approval: bool = False
    approval_reason: str = ""
