from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from rag_agent_platform.agent.graph import Agent
from rag_agent_platform.agent.offline import route
from rag_agent_platform.agent.sample_db import ensure_sample_db
from rag_agent_platform.agent.tools import build_tools
from rag_agent_platform.errors import ProviderError

from .conftest import FakeSearch, ScriptedChatModel


@pytest.fixture
def tools(tmp_path):
    return build_tools(ensure_sample_db(tmp_path / "s.sqlite3"), True, FakeSearch)


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.mark.parametrize(
    ("message", "tool"),
    [
        ("calculate (1250 * 18) / 100", "calculator"),
        ("What is 15% of 2000?", None),  # natural-language maths is not guessed offline
        ("what is 2^8?", "calculator"),
        ("revenue by category", "query_database"),
        ("Show me the top customers", "query_database"),
        ("sql: SELECT 1", "query_database"),
        ("SELECT name FROM customers", "query_database"),
        ("show database schema", "get_database_schema"),
        ("search: latest LangGraph release", "web_search"),
        ("write me a poem", None),
    ],
)
def test_offline_router(message, tool):
    reply = route(message)
    if tool is None:
        assert not reply.tool_calls and "Offline mode" in reply.content
    else:
        assert reply.tool_calls[0]["name"] == tool


def test_offline_agent_runs_graph(tools):
    result = Agent(tools, None, 6).run("revenue by category")
    assert result.mode == "offline"
    assert [s.tool for s in result.steps] == ["query_database"]
    data = json.loads(result.steps[0].output)
    assert data["columns"] == ["category", "revenue"]
    assert data["rows"][0] == ["Technology", 5097.0]


def test_offline_agent_reports_tool_errors(tools):
    result = Agent(tools, None, 6).run("sql: DELETE FROM orders")
    assert result.steps[0].status == "error"
    assert "Only SELECT" in result.answer


def test_live_agent_multi_step(tools):
    model = ScriptedChatModel(
        responses=[
            call("get_database_schema", {}, "a"),
            call("query_database", {"sql": "SELECT COUNT(*) FROM orders"}, "b"),
            call("calculator", {"expression": "16 * 2"}, "c"),
            AIMessage("There are 16 orders; doubled that is 32."),
        ],
        calls=[],
    )
    result = Agent(tools, model, 6).run(
        "How many orders, doubled?", [("user", "hi"), ("assistant", "hello")]
    )
    assert result.mode == "live"
    assert [s.tool for s in result.steps] == ["get_database_schema", "query_database", "calculator"]
    assert result.steps[1].output == '{"columns": ["COUNT(*)"], "rows": [[16]], "truncated": false}'
    assert result.steps[2].output == "32"
    assert result.answer.endswith("32.")
    # system prompt + history + new message were sent on the first call
    assert [m.type for m in model.calls[0]] == ["system", "human", "ai", "human"]


def test_live_agent_web_search(tools):
    model = ScriptedChatModel(
        responses=[call("web_search", {"query": "langgraph"}), AIMessage("See example.com/a")],
        calls=[],
    )
    result = Agent(tools, model, 6).run("search langgraph")
    assert "https://example.com/a" in result.steps[0].output


def test_step_limit_is_enforced(tools):
    loop = [call("calculator", {"expression": "1+1"}, f"id{i}") for i in range(20)]
    model = ScriptedChatModel(responses=loop, calls=[])
    with pytest.raises(ProviderError, match="did not finish within 2 tool steps"):
        Agent(tools, model, 2).run("loop forever")


def test_provider_failure_is_explicit(tools):
    model = ScriptedChatModel(responses=[], calls=[], fail_with="invalid api key")
    with pytest.raises(ProviderError, match="invalid api key"):
        Agent(tools, model, 6).run("hi")
