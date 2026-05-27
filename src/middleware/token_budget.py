from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)


class TokenBudgetExceeded(Exception):
    def __init__(self, agent: str, used: int, budget: int) -> None:
        self.agent = agent
        self.used = used
        self.budget = budget
        super().__init__(
            f"Agent '{agent}' exceeded token budget: {used}/{budget}"
        )


class TokenBudgetGuard:
    def __init__(self, model: str = "gpt-4o") -> None:
        try:
            self._encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def truncate_to_budget(self, text: str, budget: int, reserve: int = 500) -> str:
        available = budget - reserve
        if available <= 0:
            return ""
        tokens = self._encoder.encode(text)
        if len(tokens) <= available:
            return text
        logger.warning(
            "Truncating input from %d to %d tokens (budget=%d, reserve=%d)",
            len(tokens),
            available,
            budget,
            reserve,
        )
        return self._encoder.decode(tokens[:available])

    def check_budget(self, agent: str, text: str, budget: int) -> None:
        used = self.count_tokens(text)
        if used > budget:
            raise TokenBudgetExceeded(agent, used, budget)
