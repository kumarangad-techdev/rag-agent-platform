from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag_agent_platform.api.app import create_app
from rag_agent_platform.config import Settings
from rag_agent_platform.rag.embeddings import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that returns pre-scripted messages and records what it was sent."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    fail_with: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw):
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        self.calls.append(list(messages))
        reply = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FakeSearch:
    def text(self, query: str, max_results: int = 5) -> list[dict]:
        return [
            {"title": f"Result for {query}", "href": "https://example.com/a", "body": "Snippet A"},
            {"title": "Second", "href": "https://example.com/b", "body": "Snippet B"},
        ][:max_results]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        provider="offline",
        embeddings="hashing",
        data_dir=tmp_path / "data",
        samples_dir=ROOT / "sample_docs",
        static_dir=tmp_path / "no-ui",
        chunk_size=400,
        chunk_overlap=60,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(
        settings, embedder=HashingEmbedder(), search_client_factory=FakeSearch, in_memory=True
    )
    with TestClient(app) as c:
        yield c
