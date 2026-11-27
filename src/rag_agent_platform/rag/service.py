"""Document Q&A service: ingest files, retrieve top-k chunks, answer with citations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePath

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..errors import IngestError, ProviderError
from ..llm import message_text
from .chunking import chunk_text
from .embeddings import Embedder
from .loaders import file_type, load_document
from .store import Hit, StoredChunk, VectorStore

_CITATION = re.compile(r"\[S(\d+)\]")

ANSWER_PROMPT = """You answer questions using only the numbered sources provided.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every claim with its source marker, for example [S1] or [S2][S3].
- If the sources do not contain the answer, reply exactly:
  "I could not find this in the uploaded documents."
- Be concise."""


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    chunks: int
    duplicate: bool = False


@dataclass
class Source:
    marker: str
    doc_id: str
    filename: str
    chunk_index: int
    score: float
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass
class Answer:
    answer: str
    mode: str
    sources: list[Source]
    warnings: list[str] = field(default_factory=list)


class RagService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chat_model: BaseChatModel | None,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---------------------------------------------------------------- ingestion
    def ingest(self, filename: str, data: bytes) -> IngestResult:
        filename = PurePath(filename).name or "document"
        kind = file_type(filename)
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if self.store.has_document(doc_id):
            return IngestResult(doc_id=doc_id, filename=filename, chunks=0, duplicate=True)

        sections = load_document(filename, data)
        uploaded_at = datetime.now(UTC).isoformat(timespec="seconds")
        texts: list[str] = []
        metas: list[dict] = []
        for section in sections:
            for piece in chunk_text(section.text, self.chunk_size, self.chunk_overlap):
                meta = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": kind,
                    "uploaded_at": uploaded_at,
                    "chunk_index": len(texts),
                }
                if section.page is not None:
                    meta["page"] = section.page
                if section.heading:
                    meta["heading"] = section.heading
                texts.append(piece)
                metas.append(meta)
        if not texts:
            raise IngestError(f"'{filename}' produced no chunks.")

        vectors = self.embedder.embed_documents(texts)
        chunks = [
            StoredChunk(id=f"{doc_id}:{i}", text=t, metadata=m, embedding=v)
            for i, (t, m, v) in enumerate(zip(texts, metas, vectors, strict=True))
        ]
        self.store.add(chunks)
        return IngestResult(doc_id=doc_id, filename=filename, chunks=len(chunks))

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for meta in self.store.all_metadata():
            doc = docs.setdefault(
                meta["doc_id"],
                {
                    "doc_id": meta["doc_id"],
                    "filename": meta["filename"],
                    "file_type": meta.get("file_type", ""),
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "chunks": 0,
                },
            )
            doc["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def delete(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id) > 0

    # ---------------------------------------------------------------- question answering
    def retrieve(self, question: str, top_k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(question), top_k, doc_ids)

    def ask(self, question: str, top_k: int, doc_ids: list[str] | None = None) -> Answer:
        hits = self.retrieve(question, top_k, doc_ids)
        sources = [_to_source(i, hit) for i, hit in enumerate(hits, start=1)]
        if not sources:
            return Answer(
                answer="No documents are indexed yet. Upload a file or load the samples first.",
                mode=self.mode,
                sources=[],
            )
        if self.chat_model is None:
            return Answer(
                answer=(
                    "Offline mode: no language model is configured, so here is the most "
                    f"relevant passage [S1]:\n\n{sources[0].text}"
                ),
                mode=self.mode,
                sources=sources,
            )
        return self._generate(question, sources)

    @property
    def mode(self) -> str:
        return "offline" if self.chat_model is None else "live"

    def _generate(self, question: str, sources: list[Source]) -> Answer:
        context = "\n\n".join(f"[{s.marker}] ({_label(s)})\n{s.text}" for s in sources)
        messages = [
            SystemMessage(ANSWER_PROMPT),
            HumanMessage(f"Sources:\n\n{context}\n\nQuestion: {question}"),
        ]
        try:
            reply = self.chat_model.invoke(messages)
        except Exception as exc:  # provider SDKs raise many different types
            raise ProviderError(f"The model provider failed: {exc}") from exc
        text = message_text(reply).strip()
        if not text:
            raise ProviderError("The model returned an empty answer.")

        cited = {int(n) for n in _CITATION.findall(text)}
        warnings = [
            f"The answer cited [S{n}], which was not among the retrieved sources."
            for n in sorted(cited)
            if not 1 <= n <= len(sources)
        ]
        if not cited and "could not find" not in text.lower():
            warnings.append("The answer contains no source citations; verify it manually.")
        return Answer(answer=text, mode="live", sources=sources, warnings=warnings)


def _to_source(index: int, hit: Hit) -> Source:
    m = hit.metadata
    return Source(
        marker=f"S{index}",
        doc_id=m["doc_id"],
        filename=m["filename"],
        chunk_index=int(m["chunk_index"]),
        score=hit.score,
        text=hit.text,
        page=m.get("page"),
        heading=m.get("heading"),
    )


def _label(source: Source) -> str:
    where = source.filename
    if source.page is not None:
        where += f", page {source.page}"
    if source.heading:
        where += f", section '{source.heading}'"
    return where
