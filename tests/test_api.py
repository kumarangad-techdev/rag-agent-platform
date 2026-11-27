from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_agent_platform.api.app import create_app
from rag_agent_platform.config import Settings
from rag_agent_platform.errors import ConfigError
from rag_agent_platform.rag.embeddings import HashingEmbedder


def test_health(client):
    body = client.get("/api/health").json()
    assert body["mode"] == "offline"
    assert body["embeddings"] == "hashing:512"
    assert body["documents"] == 0
    assert body["offline_examples"]


def test_upload_list_ask_delete(client):
    files = [
        ("files", ("faq.txt", b"Refunds are accepted within 30 days of delivery.", "text/plain")),
        ("files", ("bad.png", b"\x89PNG", "image/png")),
    ]
    results = client.post("/api/documents", files=files).json()
    assert results[0]["chunks"] == 1 and results[0]["error"] is None
    assert "Unsupported" in results[1]["error"]

    docs = client.get("/api/documents").json()
    assert [d["filename"] for d in docs] == ["faq.txt"]

    answer = client.post("/api/qa", json={"question": "When are refunds accepted?"}).json()
    assert answer["sources"][0]["filename"] == "faq.txt"

    assert client.delete(f"/api/documents/{docs[0]['doc_id']}").status_code == 204
    assert client.delete(f"/api/documents/{docs[0]['doc_id']}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_upload_size_limit(settings):
    settings.max_upload_mb = 1
    app = create_app(settings, embedder=HashingEmbedder(), in_memory=True)
    with TestClient(app) as c:
        big = b"a " * (600 * 1024)
        result = c.post("/api/documents", files=[("files", ("big.txt", big))]).json()
    assert "larger than 1 MB" in result[0]["error"]


def test_samples_endpoint_and_duplicates(client):
    first = client.post("/api/documents/samples").json()
    assert len(first) == 3 and not any(r["duplicate"] for r in first)
    second = client.post("/api/documents/samples").json()
    assert all(r["duplicate"] for r in second)
    assert client.get("/api/health").json()["documents"] == 3


def test_qa_validation(client):
    assert client.post("/api/qa", json={"question": ""}).status_code == 422
    assert client.post("/api/qa", json={"question": "x", "top_k": 99}).status_code == 422


def test_agent_endpoint(client):
    body = client.post("/api/agent", json={"message": "calculate 12 * 12"}).json()
    assert body["answer"] == "Result: 144"
    assert body["steps"][0]["tool"] == "calculator"


def test_agent_web_search_offline(client):
    body = client.post("/api/agent", json={"message": "search: fastapi"}).json()
    assert body["steps"][0]["tool"] == "web_search"
    assert "example.com" in body["answer"]


def test_root_without_ui_explains_itself(client):
    assert "UI is not built" in client.get("/").json()["message"]


def test_serves_built_ui(settings, tmp_path):
    ui = tmp_path / "dist"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html>app</html>")
    (ui / "assets" / "app.js").write_text("console.log(1)")
    settings.static_dir = ui
    with TestClient(create_app(settings, embedder=HashingEmbedder(), in_memory=True)) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/some/client/route").text == "<html>app</html>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/api/unknown").status_code == 404


def test_live_mode_requires_model_and_key(tmp_path):
    with pytest.raises(ConfigError, match="RAG_MODEL"):
        Settings(_env_file=None, provider="openai", data_dir=tmp_path)
    with pytest.raises(ConfigError, match="RAG_API_KEY"):
        Settings(_env_file=None, provider="anthropic", model="m", data_dir=tmp_path)
