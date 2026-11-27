"""Agent tools. Each tool's core logic is a plain function so it can be tested directly."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, ToolException, tool

from ..errors import ToolInputError

# --------------------------------------------------------------------------- calculator

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPRESSION_LENGTH = 200


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic. No names, attributes or calls outside a whitelist."""
    expression = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)  # 1,250 -> 1250
    if not expression:
        raise ToolInputError("Empty expression.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolInputError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(f"Not a valid arithmetic expression: {expression}") from exc
    try:
        value = _eval(tree.body)
    except ZeroDivisionError as exc:
        raise ToolInputError("Division by zero.") from exc
    except (ValueError, OverflowError) as exc:
        raise ToolInputError(f"Math error: {exc}") from exc
    return _format_number(value)


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ToolInputError("Exponent or base too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])
    raise ToolInputError(f"Unsupported element in expression: {ast.dump(node)[:60]}")


def _format_number(value: float | int) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolInputError("Result is not a finite number.")
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


# --------------------------------------------------------------------------- database

MAX_ROWS = 50
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)
_BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile"}


def _authorizer(action: int, arg1: str | None, *_: Any) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg1 or "").lower() in _BLOCKED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ToolInputError("The sample database has not been created.")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2)
    conn.set_authorizer(_authorizer)
    return conn


def database_schema(db_path: Path) -> str:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        schema = {
            table: [f"{col[1]} {col[2]}" for col in conn.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }
    finally:
        conn.close()
    return json.dumps(schema, indent=1)


def run_select(db_path: Path, sql: str) -> str:
    """Run a single read-only SELECT. Writes, PRAGMA, ATTACH and multi-statements are refused."""
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise ToolInputError("Empty SQL.")
    if ";" in statement:
        raise ToolInputError("Only a single SQL statement is allowed.")
    first_word = statement.split(None, 1)[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise ToolInputError("Only SELECT queries are allowed; the database is read-only.")

    conn = _connect_read_only(db_path)
    budget = {"ticks": 0}

    def progress() -> int:
        budget["ticks"] += 1
        return 1 if budget["ticks"] > 500 else 0  # non-zero aborts a runaway query

    conn.set_progress_handler(progress, 10_000)
    try:
        cursor = conn.execute(statement)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(MAX_ROWS + 1)
    except sqlite3.DatabaseError as exc:
        raise ToolInputError(f"SQL error: {exc}") from exc
    finally:
        conn.close()
    truncated = len(rows) > MAX_ROWS
    return json.dumps(
        {"columns": columns, "rows": [list(r) for r in rows[:MAX_ROWS]], "truncated": truncated}
    )


# --------------------------------------------------------------------------- web search


def search_web(
    query: str, max_results: int = 5, client_factory: Callable[[], Any] | None = None
) -> str:
    query = query.strip()
    if not query:
        raise ToolInputError("Empty search query.")
    if client_factory is None:
        from ddgs import DDGS

        client_factory = DDGS
    try:
        results = client_factory().text(query, max_results=max(1, min(max_results, 8)))
    except Exception as exc:  # network errors, rate limits, parser changes
        raise ToolInputError(f"Web search failed: {exc}") from exc
    cleaned = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results or []
    ]
    if not cleaned:
        return "No web results found."
    return json.dumps(cleaned, ensure_ascii=False)


# --------------------------------------------------------------------------- LangChain wrappers


def build_tools(
    db_path: Path,
    web_search_enabled: bool,
    search_client_factory: Callable[[], Any] | None = None,
) -> list[BaseTool]:
    """Create the tool set bound to this app's configuration."""

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression exactly. Supports + - * / // % ** and
        sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max, pi, e.
        Use this for any arithmetic instead of computing it yourself."""
        return _guard(calculate, expression)

    @tool
    def get_database_schema() -> str:
        """Return the tables and columns of the bookshop SQLite database
        (customers, products, orders, order_items). Call this before writing SQL."""
        return _guard(database_schema, db_path)

    @tool
    def query_database(sql: str) -> str:
        """Run ONE read-only SQLite SELECT against the bookshop database and return JSON with
        columns and rows (max 50 rows). Writes are refused. Revenue = quantity * unit_price
        from order_items; exclude orders with status 'cancelled' unless asked."""
        return _guard(run_select, db_path, sql)

    @tool
    def web_search(query: str) -> str:
        """Search the public web (DuckDuckGo) for current information. Returns titles, URLs and
        snippets. Cite the URLs you rely on."""
        if not web_search_enabled:
            return "Web search is disabled on this server (RAG_WEB_SEARCH=false)."
        return _guard(search_web, query, 5, search_client_factory)

    tools = [calculator, get_database_schema, query_database, web_search]
    for t in tools:
        t.handle_tool_error = True  # ToolException -> clean error message for the model
    return tools


def _guard(fn: Callable[..., str], *args: Any) -> str:
    """Convert expected input errors into ToolException so the model sees a clean message."""
    try:
        return fn(*args)
    except ToolInputError as exc:
        raise ToolException(str(exc)) from exc
