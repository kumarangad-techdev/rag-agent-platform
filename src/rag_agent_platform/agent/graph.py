"""The LangGraph agent: planner -> tools -> planner ... -> END.

In live mode the planner is a chat model with the tools bound to it (a ReAct loop).
In offline mode the planner is the rule-based router from ``offline.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..errors import ProviderError
from ..llm import message_text
from .offline import offline_planner

SYSTEM_PROMPT = """You are a helpful assistant with tools.
- Use `calculator` for any arithmetic; never do maths in your head.
- For questions about the bookshop's customers, products or orders, call `get_database_schema`
  first, then `query_database` with a single SELECT.
- Use `web_search` for current or external facts, and cite the URLs you used.
- If a tool returns an error, fix the input and retry once, or explain the problem.
- Answer concisely and state which tool results your answer is based on."""

MAX_TOOL_OUTPUT_CHARS = 4000


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    output: str
    status: str


@dataclass
class AgentResult:
    answer: str
    mode: str
    steps: list[Step] = field(default_factory=list)


class Agent:
    def __init__(self, tools: list[BaseTool], chat_model: BaseChatModel | None, max_steps: int):
        self.mode = "offline" if chat_model is None else "live"
        self.max_steps = max_steps
        planner = offline_planner if chat_model is None else _live_planner(chat_model, tools)

        graph = StateGraph(MessagesState)
        graph.add_node("planner", planner)
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "planner")
        self.graph = graph.compile()

    def run(self, message: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        messages: list[BaseMessage] = []
        if self.mode == "live":
            for role, text in (history or [])[-10:]:
                messages.append(HumanMessage(text) if role == "user" else AIMessage(text))
        messages.append(HumanMessage(message))
        try:
            final = self.graph.invoke(
                {"messages": messages}, config={"recursion_limit": 2 * self.max_steps + 1}
            )
        except GraphRecursionError as exc:
            raise ProviderError(
                f"The agent did not finish within {self.max_steps} tool steps."
            ) from exc
        new_messages = final["messages"][len(messages) :]
        return AgentResult(
            answer=message_text(new_messages[-1]).strip() or "(no answer)",
            mode=self.mode,
            steps=_steps(new_messages),
        )


def _live_planner(chat_model: BaseChatModel, tools: list[BaseTool]):
    bound = chat_model.bind_tools(tools)

    def planner(state: MessagesState) -> dict:
        try:
            reply = bound.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        except Exception as exc:  # provider SDKs raise many different types
            raise ProviderError(f"The model provider failed: {exc}") from exc
        return {"messages": [reply]}

    return planner


def _steps(messages: list[BaseMessage]) -> list[Step]:
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    steps: list[Step] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call["id"])
            output = message_text(result) if result else "(no result)"
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + "… [truncated]"
            steps.append(
                Step(
                    tool=call["name"],
                    args=json.loads(json.dumps(call["args"], default=str)),
                    output=output,
                    status=(result.status if result else "error") or "success",
                )
            )
    return steps
