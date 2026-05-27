from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    def __init__(self, agent: str, attempts: int, last_error: Exception) -> None:
        self.agent = agent
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Agent '{agent}' failed after {attempts} attempts: {last_error}"
        )


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    agent_name: str = "unknown",
    max_retries: int | None = None,
    base_delay: float | None = None,
    **kwargs: Any,
) -> T:
    retries = max_retries if max_retries is not None else settings.max_retries
    delay = base_delay if base_delay is not None else settings.retry_base_delay
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = delay * (2 ** (attempt - 1))
            logger.warning(
                "Agent %s attempt %d/%d failed (%s), retrying in %.1fs",
                agent_name,
                attempt,
                retries,
                exc,
                wait,
            )
            await asyncio.sleep(wait)

    raise RetryExhausted(agent_name, retries, last_exc)  # type: ignore[arg-type]
