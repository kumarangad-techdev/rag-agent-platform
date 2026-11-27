"""Offline planner: a transparent rule-based router used when no language model is configured.

It produces the same tool-call messages a model would, so the LangGraph graph, the ToolNode and
the API response are identical in both modes. It only understands the patterns listed in
``OFFLINE_EXAMPLES``; anything else gets an explicit limitation message, never a guess.
"""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..llm import message_text

CANNED_QUERIES: dict[str, tuple[str, str]] = {
    "revenue by category": (
        "Revenue by product category (excluding cancelled orders)",
        "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "JOIN orders o ON o.id = oi.order_id WHERE o.status != 'cancelled' "
        "GROUP BY p.category ORDER BY revenue DESC",
    ),
    "top customers": (
        "Top customers by spend (excluding cancelled orders)",
        "SELECT c.name, c.city, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS spent "
        "FROM customers c JOIN orders o ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' "
        "GROUP BY c.id ORDER BY spent DESC LIMIT 5",
    ),
    "orders by status": (
        "Number of orders by status",
        "SELECT status, COUNT(*) AS orders FROM orders GROUP BY status ORDER BY orders DESC",
    ),
    "best selling books": (
        "Best-selling books by units sold",
        "SELECT p.title, SUM(oi.quantity) AS units FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id "
        "WHERE o.status != 'cancelled' GROUP BY p.id ORDER BY units DESC LIMIT 5",
    ),
}

OFFLINE_EXAMPLES = [
    "calculate (1250 * 18) / 100",
    "sqrt(144) + 2^5",
    "revenue by category",
    "top customers",
    "orders by status",
    "best selling books",
    "show database schema",
    "sql: SELECT city, COUNT(*) FROM customers GROUP BY city",
    "search: LangGraph documentation",
]

_MATH_CHARS = r"[\d\s.,+\-*/%^()×÷]*"
_MATH_WORDS = r"(?:sqrt|log10|log|exp|sin|cos|tan|abs|round|min|max|pi|e)"
_MATH = re.compile(rf"^{_MATH_CHARS}(?:{_MATH_WORDS}{_MATH_CHARS})*$")
_MATH_PREFIX = re.compile(
    r"^(?:please\s+)?(?:calculate|compute|evaluate|what\s+is|what's|solve)\s*:?\s*", re.I
)
_SEARCH_PREFIX = re.compile(r"^(?:search|web|search the web for|look up|google)\s*:?\s+", re.I)
_SQL_PREFIX = re.compile(r"^sql\s*:\s*", re.I)


def _call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"offline-{uuid.uuid4().hex[:8]}"}],
    )


def route(text: str) -> AIMessage:
    """Map one user message to a tool call, or to an explicit limitation message."""
    text = text.strip()
    lowered = text.lower().rstrip("?. ")

    if _SQL_PREFIX.match(text):
        return _call("query_database", {"sql": _SQL_PREFIX.sub("", text)})
    if re.match(r"^(select|with)\s", text, re.I):
        return _call("query_database", {"sql": text})
    for key, (_, sql) in CANNED_QUERIES.items():
        if key in lowered:
            return _call("query_database", {"sql": sql})
    if any(word in lowered for word in ("schema", "tables", "columns")):
        return _call("get_database_schema", {})
    if _SEARCH_PREFIX.match(text):
        return _call("web_search", {"query": _SEARCH_PREFIX.sub("", text)})

    expression = _MATH_PREFIX.sub("", text).rstrip("?= ")
    if expression and re.search(r"\d|pi", expression) and _MATH.match(expression.lower()):
        return _call("calculator", {"expression": expression})

    examples = "\n".join(f"• {e}" for e in OFFLINE_EXAMPLES)
    return AIMessage(
        content=(
            "Offline mode has no language model, so it only understands these patterns:\n"
            f"{examples}\n\nConfigure RAG_PROVIDER, RAG_MODEL and RAG_API_KEY for free-form "
            "questions."
        )
    )


def summarise(tool_message: ToolMessage) -> AIMessage:
    """Turn a tool result into the final answer without a model."""
    content = message_text(tool_message)
    if tool_message.status == "error" or content.startswith("Error"):
        return AIMessage(content=f"The {tool_message.name} tool could not complete: {content}")
    if tool_message.name == "calculator":
        return AIMessage(content=f"Result: {content}")
    if tool_message.name == "query_database":
        data = json.loads(content)
        note = " (first 50 rows)" if data.get("truncated") else ""
        return AIMessage(content=f"The query returned {len(data['rows'])} row(s){note}.")
    if tool_message.name == "web_search":
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            return AIMessage(content=content)
        lines = [f"• {r['title']} — {r['url']}" for r in results]
        return AIMessage(content="Top web results:\n" + "\n".join(lines))
    return AIMessage(content=content)


def offline_planner(state: dict) -> dict:
    messages: list[BaseMessage] = state["messages"]
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return {"messages": [summarise(last)]}
    if isinstance(last, HumanMessage):
        return {"messages": [route(message_text(last))]}
    return {"messages": [AIMessage(content="Nothing to do.")]}
