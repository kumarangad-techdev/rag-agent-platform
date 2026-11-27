"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    mode: Literal["offline", "live"]
    provider: str
    model: str | None
    embeddings: str
    documents: int
    chunks: int
    web_search: bool
    offline_examples: list[str]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    uploaded_at: str
    chunks: int


class IngestResponse(BaseModel):
    doc_id: str | None = None
    filename: str
    chunks: int = 0
    duplicate: bool = False
    error: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    doc_ids: list[str] | None = None


class SourceOut(BaseModel):
    marker: str
    doc_id: str
    filename: str
    chunk_index: int
    score: float
    text: str
    page: int | None = None
    heading: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    mode: Literal["offline", "live"]
    sources: list[SourceOut]
    warnings: list[str] = []


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)


class StepOut(BaseModel):
    tool: str
    args: dict[str, Any]
    output: str
    status: str


class AgentResponse(BaseModel):
    answer: str
    mode: Literal["offline", "live"]
    steps: list[StepOut]
