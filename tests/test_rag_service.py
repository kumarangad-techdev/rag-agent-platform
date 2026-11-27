from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from rag_agent_platform.errors import ConfigError, ProviderError
from rag_agent_platform.rag.embeddings import HashingEmbedder, HuggingFaceEmbedder
from rag_agent_platform.rag.service import RagService
from rag_agent_platform.rag.store import VectorStore

from .conftest import ROOT, ScriptedChatModel


def make_service(chat_model=None, path=None, embedder=None) -> RagService:
    embedder = embedder or HashingEmbedder()
    return RagService(VectorStore(path, embedder.name), embedder, chat_model, 400, 60)


def load_samples(service: RagService) -> None:
    for path in sorted((ROOT / "sample_docs").iterdir()):
        service.ingest(path.name, path.read_bytes())


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed_query("annual leave days"), e.embed_query("annual leave days")
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_huggingface_embedder_uses_sentence_transformers(monkeypatch):
    class FakeModel:
        def __init__(self, name, device):
            self.name = name

        def encode(self, texts, normalize_embeddings, batch_size):
            assert normalize_embeddings
            return np.array([[1.0, 0.0]] * len(texts))

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.name == "huggingface:sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_missing_sentence_transformers_fails_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigError, match="sentence-transformers is not installed"):
        HuggingFaceEmbedder("any")


def test_ingest_list_and_delete():
    service = make_service()
    load_samples(service)
    docs = {d["filename"]: d for d in service.list_documents()}
    assert set(docs) == {"employee_handbook.md", "orbit_lamp_faq.txt", "rag_primer.pdf"}
    assert all(d["chunks"] > 0 for d in docs.values())
    assert service.delete(docs["orbit_lamp_faq.txt"]["doc_id"])
    assert len(service.list_documents()) == 2
    assert not service.delete("missing")


def test_duplicate_upload_is_detected():
    service = make_service()
    data = b"Some unique text about refunds."
    first = service.ingest("a.txt", data)
    second = service.ingest("copy-of-a.txt", data)
    assert not first.duplicate and second.duplicate
    assert second.doc_id == first.doc_id
    assert len(service.list_documents()) == 1


def test_retrieval_finds_the_right_document():
    service = make_service()
    load_samples(service)
    hits = service.retrieve("How long does the lamp battery last at full brightness?", 3)
    assert hits[0].metadata["filename"] == "orbit_lamp_faq.txt"
    assert hits[0].score >= hits[-1].score


def test_retrieval_can_be_limited_to_documents():
    service = make_service()
    load_samples(service)
    pdf = next(d for d in service.list_documents() if d["filename"] == "rag_primer.pdf")
    hits = service.retrieve("battery life", 5, doc_ids=[pdf["doc_id"]])
    assert hits and {h.metadata["filename"] for h in hits} == {"rag_primer.pdf"}


def test_offline_answer_is_retrieval_only():
    service = make_service()
    load_samples(service)
    answer = service.ask("How many days of paid annual leave?", 3)
    assert answer.mode == "offline"
    assert answer.answer.startswith("Offline mode")
    assert answer.sources[0].heading == "Leave policy"
    assert "24 days" in answer.sources[0].text


def test_no_documents_message():
    answer = make_service().ask("anything", 3)
    assert answer.sources == [] and "No documents" in answer.answer


def test_live_answer_checks_citations():
    model = ScriptedChatModel(
        responses=[AIMessage("Employees get 24 days [S1]. Also see [S9].")], calls=[]
    )
    service = make_service(chat_model=model)
    load_samples(service)
    answer = service.ask("How many days of paid annual leave?", 3)
    assert answer.mode == "live"
    assert answer.warnings == ["The answer cited [S9], which was not among the retrieved sources."]
    prompt = model.calls[0][1].content
    assert "[S1]" in prompt and "Question: How many days" in prompt


def test_live_answer_without_citations_is_flagged():
    model = ScriptedChatModel(responses=[AIMessage("Twenty-four.")], calls=[])
    service = make_service(chat_model=model)
    load_samples(service)
    assert "no source citations" in service.ask("leave?", 2).warnings[0]


def test_provider_failure_is_explicit():
    model = ScriptedChatModel(responses=[], calls=[], fail_with="rate limited")
    service = make_service(chat_model=model)
    load_samples(service)
    with pytest.raises(ProviderError, match="rate limited"):
        service.ask("leave?", 2)


def test_embedder_mismatch_is_refused(tmp_path):
    service = make_service(path=tmp_path / "chroma")
    service.ingest("a.txt", b"hello world")

    class OtherEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__(256)

    with pytest.raises(ConfigError, match="cannot be compared"):
        VectorStore(tmp_path / "chroma", OtherEmbedder().name)


def test_store_persists_across_restarts(tmp_path):
    make_service(path=tmp_path / "chroma").ingest("a.txt", b"persistent fact about otters")
    assert len(make_service(path=tmp_path / "chroma").list_documents()) == 1
