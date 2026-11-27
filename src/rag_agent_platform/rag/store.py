"""ChromaDB persistence for chunks. Embeddings are computed by us, never by Chroma."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..errors import ConfigError

COLLECTION = "documents"


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, 1.0 = identical direction


class VectorStore:
    def __init__(self, path: Path | None, embedder_name: str) -> None:
        settings = ChromaSettings(anonymized_telemetry=False)
        name = COLLECTION
        if path is None:
            # Ephemeral clients share one in-process system, so isolate each store by name.
            self._client = chromadb.EphemeralClient(settings=settings)
            name = f"{COLLECTION}-{uuid.uuid4().hex[:12]}"
        else:
            self._client = chromadb.PersistentClient(path=str(path), settings=settings)
        self._collection = self._client.get_or_create_collection(
            name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": embedder_name},
        )
        stored = (self._collection.metadata or {}).get("embedder")
        if stored != embedder_name and self._collection.count() > 0:
            raise ConfigError(
                f"The vector store was built with '{stored}' but the app is configured for "
                f"'{embedder_name}'. Vectors from different embedders cannot be compared: "
                "delete the data directory or switch back."
            )
        self._lock = threading.Lock()

    def has_document(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return bool(found["ids"])

    def add(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return
        with self._lock:
            self._collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
                embeddings=[c.embedding for c in chunks],
            )

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)

    def all_metadata(self) -> list[dict[str, Any]]:
        return list(self._collection.get(include=["metadatas"])["metadatas"] or [])

    def count(self) -> int:
        return self._collection.count()

    def search(
        self, embedding: list[float], top_k: int, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        where: dict[str, Any] | None = None
        if doc_ids:
            where = {"doc_id": {"$in": doc_ids}}
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(id=i, text=t, metadata=dict(m), score=round(1.0 - float(d), 4))
            for i, t, m, d in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
