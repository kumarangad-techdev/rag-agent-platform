"""Embedding back-ends.

``HuggingFaceEmbedder`` is the real one (sentence-transformers, runs locally on CPU).
``HashingEmbedder`` is a deterministic lexical embedder with no downloads; it exists so tests
and CI can exercise the whole pipeline without fetching a model. It is not semantic.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from ..config import Settings
from ..errors import ConfigError

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "sentence-transformers is not installed. Install it, or set "
                "RAG_EMBEDDINGS=hashing for the offline test embedder."
            ) from exc
        self.name = f"huggingface:{model_name}"
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
        except OSError as exc:
            raise ConfigError(
                f"Could not load embedding model '{model_name}'. The first run needs internet "
                f"access to download it from Hugging Face. ({exc})"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbedder:
    """Feature-hashed unigrams and bigrams, L2-normalised. Deterministic across runs."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self.name = f"hashing:{dimensions}"

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embeddings == "hashing":
        return HashingEmbedder()
    return HuggingFaceEmbedder(settings.embedding_model)
