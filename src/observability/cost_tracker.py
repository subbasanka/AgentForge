from __future__ import annotations

from src.config import settings
from src.models.state import AgentRole, CostEntry


class CostTracker:
    def __init__(
        self,
        input_cost_per_m: float | None = None,
        output_cost_per_m: float | None = None,
    ) -> None:
        self._input_cost_per_m = input_cost_per_m or settings.input_cost_per_m
        self._output_cost_per_m = output_cost_per_m or settings.output_cost_per_m

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self._input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * self._output_cost_per_m
        return round(input_cost + output_cost, 6)

    def record(
        self,
        agent: AgentRole,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEntry:
        return CostEntry(
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.compute_cost(input_tokens, output_tokens),
        )

    @staticmethod
    def aggregate(entries: list[CostEntry]) -> float:
        return round(sum(e.cost_usd for e in entries), 6)
