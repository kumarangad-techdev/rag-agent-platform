"""HTTP API and static UI hosting."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agent.graph import Agent
from ..agent.offline import OFFLINE_EXAMPLES
from ..agent.sample_db import ensure_sample_db
from ..agent.tools import build_tools
from ..config import Settings
from ..errors import ConfigError, IngestError, ProviderError, ToolInputError
from ..llm import build_chat_model
from ..rag.embeddings import Embedder, build_embedder
from ..rag.loaders import SUPPORTED_EXTENSIONS
from ..rag.service import RagService
from ..rag.store import VectorStore
from .schemas import (
    AgentRequest,
    AgentResponse,
    AnswerResponse,
    DocumentInfo,
    Health,
    IngestResponse,
    QuestionRequest,
)


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chat_model=None,
    search_client_factory=None,
    in_memory: bool = False,
) -> FastAPI:
    """Build the app. Tests inject an embedder, a fake chat model and a fake search client."""
    settings = settings or Settings()
    settings.ensure_dirs()
    embedder = embedder or build_embedder(settings)
    chat_model = chat_model if chat_model is not None else build_chat_model(settings)

    store = VectorStore(None if in_memory else settings.chroma_dir, embedder.name)
    rag = RagService(store, embedder, chat_model, settings.chunk_size, settings.chunk_overlap)
    db_path = ensure_sample_db(settings.sample_db_path)
    tools = build_tools(db_path, settings.web_search, search_client_factory)
    agent = Agent(tools, chat_model, settings.agent_max_steps)

    app = FastAPI(title="RAG Agent Platform", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.state.settings, app.state.rag, app.state.agent = settings, rag, agent

    @app.exception_handler(ProviderError)
    async def _provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        docs = rag.list_documents()
        return Health(
            mode=rag.mode,
            provider=settings.provider,
            model=settings.model,
            embeddings=embedder.name,
            documents=len(docs),
            chunks=store.count(),
            web_search=settings.web_search,
            offline_examples=OFFLINE_EXAMPLES,
        )

    @app.get("/api/documents", response_model=list[DocumentInfo])
    def list_documents() -> list[dict]:
        return rag.list_documents()

    @app.post("/api/documents", response_model=list[IngestResponse])
    async def upload(files: list[UploadFile] = File(...)) -> list[IngestResponse]:
        limit = settings.max_upload_mb * 1024 * 1024
        results: list[IngestResponse] = []
        for upload_file in files:
            name = upload_file.filename or "document"
            data = await upload_file.read(limit + 1)
            if len(data) > limit:
                results.append(
                    IngestResponse(
                        filename=name, error=f"File is larger than {settings.max_upload_mb} MB."
                    )
                )
                continue
            try:
                results.append(IngestResponse(**asdict(rag.ingest(name, data))))
            except IngestError as exc:
                results.append(IngestResponse(filename=name, error=str(exc)))
        return results

    @app.post("/api/documents/samples", response_model=list[IngestResponse])
    def load_samples() -> list[IngestResponse]:
        folder = settings.samples_dir
        if not folder.is_dir():
            raise HTTPException(404, f"Sample folder '{folder}' not found.")
        results = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    results.append(
                        IngestResponse(**asdict(rag.ingest(path.name, path.read_bytes())))
                    )
                except IngestError as exc:
                    results.append(IngestResponse(filename=path.name, error=str(exc)))
        return results

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        if not rag.delete(doc_id):
            raise HTTPException(404, "Document not found.")

    @app.post("/api/qa", response_model=AnswerResponse)
    def ask(request: QuestionRequest) -> dict:
        answer = rag.ask(request.question, request.top_k or settings.top_k, request.doc_ids)
        return asdict(answer)

    @app.post("/api/agent", response_model=AgentResponse)
    def run_agent(request: AgentRequest) -> dict:
        try:
            result = agent.run(request.message, [(t.role, t.content) for t in request.history])
        except ToolInputError as exc:
            raise HTTPException(422, str(exc)) from exc
        return asdict(result)

    _mount_ui(app, settings.static_dir)
    return app


def _mount_ui(app: FastAPI, static_dir: Path) -> None:
    index = static_dir / "index.html"
    if not index.exists():

        @app.get("/", include_in_schema=False)
        def no_ui() -> dict:
            return {
                "message": "API is running. The React UI is not built: run `npm run build` in "
                "frontend/, or use the dev server on port 5173. API docs: /docs"
            }

        return

    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found.")
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)
