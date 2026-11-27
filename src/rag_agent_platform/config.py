"""Application settings, read from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Language model -------------------------------------------------------------
    # "offline" runs without any model: retrieval-only answers and a rule-based agent router.
    provider: Literal["offline", "openai", "anthropic"] = "offline"
    model: str | None = None
    api_key: SecretStr | None = None

    # --- Embeddings ------------------------------------------------------------------
    # "huggingface" uses sentence-transformers (downloads the model once).
    # "hashing" is a dependency-free lexical embedder used for tests and CI.
    embeddings: Literal["huggingface", "hashing"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval -------------------------------------------------------------------
    chunk_size: int = Field(800, ge=200, le=4000)
    chunk_overlap: int = Field(120, ge=0, le=1000)
    top_k: int = Field(4, ge=1, le=20)

    # --- Agent -----------------------------------------------------------------------
    web_search: bool = True
    agent_max_steps: int = Field(6, ge=1, le=20)

    # --- Storage and serving ---------------------------------------------------------
    data_dir: Path = Path(".ragdata")
    samples_dir: Path = Path("sample_docs")
    static_dir: Path = Path("frontend/dist")
    max_upload_mb: int = Field(10, ge=1, le=100)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
        if self.provider != "offline":
            if not self.model:
                raise ConfigError(f"RAG_MODEL is required when RAG_PROVIDER={self.provider}.")
            if not self.api_key or not self.api_key.get_secret_value():
                raise ConfigError(f"RAG_API_KEY is required when RAG_PROVIDER={self.provider}.")
        return self

    @property
    def is_live(self) -> bool:
        return self.provider != "offline"

    @cached_property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @cached_property
    def sample_db_path(self) -> Path:
        return self.data_dir / "sample_store.sqlite3"

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
