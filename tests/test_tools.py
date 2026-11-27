from __future__ import annotations

import json
import sqlite3

import pytest

from rag_agent_platform.agent.sample_db import ensure_sample_db
from rag_agent_platform.agent.tools import (
    MAX_ROWS,
    calculate,
    database_schema,
    run_select,
    search_web,
)
from rag_agent_platform.errors import ToolInputError

from .conftest import FakeSearch


@pytest.fixture
def db(tmp_path):
    return ensure_sample_db(tmp_path / "store.sqlite3")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(1250 * 18) / 100", "225"),
        ("2^10", "1024"),
        ("sqrt(144) + abs(-3)", "15"),
        ("1,250 + 750", "2000"),
        ("10 / 4", "2.5"),
        ("round(pi, 3)", "3.142"),
    ],
)
def test_calculator(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "(1).__class__",
        "9 ** 9999",
        "1 / 0",
        "sqrt(-1)",
        "",
        "x" * 300,
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expression):
    with pytest.raises(ToolInputError):
        calculate(expression)


def test_schema_lists_tables(db):
    schema = json.loads(database_schema(db))
    assert set(schema) == {"customers", "products", "orders", "order_items"}


def test_select_returns_rows(db):
    result = json.loads(run_select(db, "SELECT COUNT(*) AS n FROM customers;"))
    assert result == {"columns": ["n"], "rows": [[8]], "truncated": False}


def test_select_is_truncated(db):
    sql = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 500) "
        "SELECT x FROM n"
    )
    result = json.loads(run_select(db, sql))
    assert len(result["rows"]) == MAX_ROWS and result["truncated"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE products SET price = 0",
        "DROP TABLE customers",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT 1; DELETE FROM orders",
        "WITH t AS (SELECT 1) DELETE FROM orders",
        "SELECT load_extension('evil')",
    ],
)
def test_writes_and_tricks_are_refused(db, sql):
    with pytest.raises(ToolInputError):
        run_select(db, sql)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 16


def test_runaway_query_is_aborted(db):
    sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    with pytest.raises(ToolInputError, match="interrupted"):
        run_select(db, sql)


def test_web_search_formats_results():
    results = json.loads(search_web("langgraph", client_factory=FakeSearch))
    assert results[0] == {
        "title": "Result for langgraph",
        "url": "https://example.com/a",
        "snippet": "Snippet A",
    }


def test_web_search_failure_is_explicit():
    class Broken:
        def text(self, *a, **k):
            raise ConnectionError("offline")

    with pytest.raises(ToolInputError, match="Web search failed: offline"):
        search_web("anything", client_factory=Broken)
