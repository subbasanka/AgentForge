from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.models.state import AgentRole, PipelineState

logger = logging.getLogger(__name__)


async def approval_gate_node(state: PipelineState) -> dict:
    """Human-in-the-loop checkpoint.

    When the graph reaches this node, it pauses execution.
    The FastAPI layer surfaces the approval request, and when
    the human responds, the graph resumes from this node.
    """
    if state.approved is True:
        logger.info("Approval granted for task %s", state.task_id)
        
        # Find the last AIMessage with tool_calls in message history
        pending_tool_call = None
        for msg in reversed(state.messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                pending_tool_call = msg.tool_calls[0]
                break

        messages = [AIMessage(content="Approval gate: APPROVED — resuming pipeline.")]
        if pending_tool_call and pending_tool_call.get("name") == "file_write_tool":
            tool_args = pending_tool_call.get("args", {})
            logger.info("Executing approved file_write_tool: %s", tool_args)
            try:
                from src.tools.web_search import file_write_tool
                from langchain_core.messages import ToolMessage
                result = await file_write_tool.ainvoke(tool_args)
                logger.info("Approved tool executed successfully: %s", result)
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=pending_tool_call.get("id", ""),
                    name="file_write_tool",
                ))
            except Exception as exc:
                logger.error("Failed to execute approved tool: %s", exc)

        return {
            "pending_approval": False,
            "approved": None,
            "approval_context": "",
            "next_agent": AgentRole.SUPERVISOR,
            "messages": messages,
        }
    elif state.approved is False:
        logger.info("Approval denied for task %s", state.task_id)
        return {
            "pending_approval": False,
            "approved": None,
            "approval_context": "",
            "next_agent": AgentRole.SUPERVISOR,
            "errors": [f"Human rejected action: {state.approval_context}"],
            "messages": [AIMessage(content="Approval gate: DENIED — action skipped.")],
        }
    else:
        logger.info("Waiting for human approval on task %s: %s", state.task_id, state.approval_context)
        return {
            "pending_approval": True,
            "status": "awaiting_approval",
        }
