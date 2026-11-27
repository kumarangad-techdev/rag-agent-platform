"""End-to-end smoke test in offline mode: no API key, no network, no model download.

Usage: python scripts/smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_agent_platform.api.app import create_app
from rag_agent_platform.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,
            provider="offline",
            embeddings="hashing",
            data_dir=Path(tmp),
            samples_dir=ROOT / "sample_docs",
            web_search=False,
        )
        with TestClient(create_app(settings)) as client:
            check("health endpoint", client.get("/api/health").json()["mode"] == "offline")

            loaded = client.post("/api/documents/samples").json()
            check("3 sample documents indexed", sum(1 for r in loaded if r["chunks"]) == 3)

            answer = client.post(
                "/api/qa", json={"question": "How long is parental leave?", "top_k": 3}
            ).json()
            top = answer["sources"][0]
            check("Q&A retrieves the handbook", top["filename"] == "employee_handbook.md")

            maths = client.post("/api/agent", json={"message": "calculate (1250 * 18) / 100"})
            check("agent calculator", maths.json()["answer"] == "Result: 225")

            sql = client.post("/api/agent", json={"message": "orders by status"}).json()
            rows = json.loads(sql["steps"][0]["output"])["rows"]
            check("agent database query", ["delivered", 12] in rows)

            blocked = client.post("/api/agent", json={"message": "sql: DELETE FROM orders"})
            check("agent refuses writes", blocked.json()["steps"][0]["status"] == "error")

            search = client.post("/api/agent", json={"message": "search: fastapi"}).json()
            check("web search can be disabled", "disabled" in search["steps"][0]["output"])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
