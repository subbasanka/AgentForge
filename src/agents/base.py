from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import settings
from src.models.state import AgentRole, CostEntry, PipelineState
from src.middleware.token_budget import TokenBudgetGuard
from src.observability.cost_tracker import CostTracker
from src.observability.langfuse_client import trace_agent_call
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_token_guard = TokenBudgetGuard()
_cost_tracker = CostTracker()
_tool_registry = ToolRegistry()


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=4096,
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )


def get_token_budget(role: AgentRole) -> int:
    budgets = {
        AgentRole.SUPERVISOR: settings.supervisor_token_budget,
        AgentRole.RESEARCHER: settings.researcher_token_budget,
        AgentRole.ANALYST: settings.analyst_token_budget,
        AgentRole.WRITER: settings.writer_token_budget,
        AgentRole.REVIEWER: settings.reviewer_token_budget,
    }
    return budgets.get(role, 8000)


async def run_agent_llm(
    role: AgentRole,
    system_prompt: str,
    user_message: str,
    state: PipelineState,
    bind_tools: bool = False,
) -> dict[str, Any]:
    """Common LLM invocation logic with budget guard, cost tracking, and tracing."""
    budget = get_token_budget(role)
    truncated_context = _token_guard.truncate_to_budget(user_message, budget)

    llm = get_llm()
    tools = _tool_registry.get_tools_for_agent(role)
    if bind_tools and tools:
        llm = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=truncated_context),
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    tool_results: list[dict[str, Any]] = []

    async with trace_agent_call(state.task_id, role) as ctx:
        max_turns = 5
        turn = 0
        response = None

        while turn < max_turns:
            response = await llm.ainvoke(messages)

            input_tokens = response.usage_metadata.get("input_tokens", 0) if response.usage_metadata else 0
            output_tokens = response.usage_metadata.get("output_tokens", 0) if response.usage_metadata else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            tool_calls_raw = getattr(response, "tool_calls", []) or []
            if not tool_calls_raw:
                break

            # Check if any tool call requires human approval.
            # If so, we must STOP the loop and return so the approval gate handles it.
            has_approval_required = any(
                _tool_registry.requires_approval(tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""))
                for tc in tool_calls_raw
            )
            if has_approval_required:
                for tc in tool_calls_raw:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tool_results.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "requires_approval": True,
                    })
                break

            tool_messages = []
            for tc in tool_calls_raw:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

                matching_tools = [t for t in tools if t.name == tool_name]
                if matching_tools:
                    try:
                        result_str = await matching_tools[0].ainvoke(tool_args)
                        tool_messages.append(ToolMessage(
                            content=str(result_str),
                            tool_call_id=tool_id,
                            name=tool_name,
                        ))
                        tool_results.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": str(result_str),
                            "success": True,
                        })
                    except Exception as exc:
                        tool_messages.append(ToolMessage(
                            content=f"Error: {exc}",
                            tool_call_id=tool_id,
                            name=tool_name,
                        ))
                        tool_results.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": str(exc),
                            "success": False,
                        })

            # Append assistant's turn and the tool outputs to conversational context
            messages.append(response)
            messages.extend(tool_messages)
            turn += 1

        cost_entry = _cost_tracker.record(role, total_input_tokens, total_output_tokens)

        ctx["span"].update(
            metadata={
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cost_usd": cost_entry.cost_usd,
            }
        )

    content = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("Agent LLM Response (%s): content_len=%d, tool_calls=%s", role.value, len(content), getattr(response, 'tool_calls', []))

    return {
        "content": content,
        "cost_entry": cost_entry,
        "tool_calls": tool_results,
        "response": response,
    }
