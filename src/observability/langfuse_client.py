from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from src.config import settings
from src.models.state import AgentRole

logger = logging.getLogger(__name__)

_langfuse_instance: Any = None
_langfuse_enabled: bool = False


def get_langfuse() -> Any:
    global _langfuse_instance, _langfuse_enabled
    if _langfuse_instance is None:
        _langfuse_enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        if _langfuse_enabled:
            try:
                from langfuse import Langfuse

                _langfuse_instance = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception as exc:
                logger.warning("Failed to initialize Langfuse: %s", exc)
                _langfuse_enabled = False
                _langfuse_instance = _NoOpLangfuse()
        else:
            _langfuse_instance = _NoOpLangfuse()
    return _langfuse_instance


class _NoOpSpan:
    def span(self, **kwargs: Any) -> _NoOpSpan:
        return self

    def end(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class _NoOpLangfuse:
    def trace(self, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_observation(self, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@asynccontextmanager
async def trace_agent_call(
    task_id: str,
    agent: AgentRole,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    lf = get_langfuse()

    span = None
    manager = None
    if _langfuse_enabled and hasattr(lf, "start_as_current_observation"):
        try:
            manager = lf.start_as_current_observation(
                name=f"agent:{agent.value}",
                metadata={"task_id": task_id, "agent": agent.value, **(metadata or {})},
            )
            span = manager.__enter__()
        except Exception:
            span = _NoOpSpan()
            manager = None
    elif hasattr(lf, "trace"):
        span = lf.trace(
            id=trace_id or f"{task_id}-{agent.value}-{int(time.time() * 1000)}",
            name=f"agent:{agent.value}",
            metadata={"task_id": task_id, "agent": agent.value, **(metadata or {})},
        )
    else:
        span = _NoOpSpan()

    context: dict[str, Any] = {
        "trace": span,
        "span": span,
        "start_time": time.monotonic(),
    }
    try:
        yield context
    except Exception as exc:
        if manager is not None:
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
        else:
            _safe_end_span(span, error=str(exc))
        raise
    else:
        elapsed = time.monotonic() - context["start_time"]
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass
        else:
            _safe_end_span(span, duration_s=round(elapsed, 3))
    finally:
        try:
            lf.flush()
        except Exception:
            pass


@asynccontextmanager
async def trace_tool_call(
    parent_span: Any,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    if hasattr(parent_span, "span"):
        try:
            span = parent_span.span(
                name=f"tool:{tool_name}",
                input=arguments or {},
                metadata={"tool": tool_name},
            )
        except Exception:
            span = _NoOpSpan()
    else:
        span = _NoOpSpan()

    context: dict[str, Any] = {"span": span, "start_time": time.monotonic()}
    try:
        yield context
    except Exception as exc:
        _safe_end_span(span, error=str(exc))
        raise
    else:
        elapsed = time.monotonic() - context["start_time"]
        _safe_end_span(span, duration_ms=round(elapsed * 1000, 1))


def _safe_end_span(span: Any, **kwargs: Any) -> None:
    try:
        if hasattr(span, "end"):
            span.end(**kwargs)
    except Exception:
        pass
