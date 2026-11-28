"""API FastAPI de l'orchestrateur multi-agents (port 8002).

Graphe LangGraph : Analyste → Cardiologue RAG → Auditeur.
Les sorties sont validées par Pydantic (``AnalyzeResponse``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.clients import LLM, OllamaClient, QdrantRetriever, Retriever
from agents.config import Settings
from agents.nodes import evaluate_prompt
from agents.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    PromptCheckRequest,
    PromptCheckResponse,
)
from agents.service import Orchestrator

logger = logging.getLogger("agent-orchestrator")


def create_app(
    retriever: Retriever | None = None,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Fabrique l'application avec dépendances injectables (tests)."""
    settings = settings or Settings.from_env()
    retriever = retriever or QdrantRetriever(
        url=settings.qdrant_url,
        collection=settings.collection,
        model_name=settings.embedding_model,
        min_score=settings.min_score,
    )
    llm = llm or OllamaClient(
        url=settings.llm_url,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_num_predict,
    )
    orchestrator = Orchestrator(retriever=retriever, llm=llm, settings=settings)

    app = FastAPI(
        title="CardioAgent-Ops — agent-orchestrator",
        description="Graphe LangGraph Analyste → Cardiologue RAG → Auditeur, avec garde-fous Pydantic.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.state.orchestrator = orchestrator
    app.state.settings = settings

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "agent-orchestrator",
            "status": "ok",
            "graph": ["analyste", "cardiologue_rag", "auditeur"],
            "endpoints": ["/health", "/check_prompt", "/analyze", "/analyze/stream", "/docs"],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        services: dict[str, Any] = {}
        for name, dependency in (("qdrant", retriever), ("llm", llm)):
            try:
                services[name] = dependency.health()
            except Exception as error:
                services[name] = {"status": "down", "error": f"{type(error).__name__}: {error}"}
        healthy = all(item.get("status") == "ok" for item in services.values())
        return {
            "status": "ok" if healthy else "degraded",
            "services": services,
            "model": settings.llm_model,
            "collection": settings.collection,
        }

    @app.post("/check_prompt", response_model=PromptCheckResponse)
    def check_prompt(payload: PromptCheckRequest) -> PromptCheckResponse:
        """Première étape du workflow : valide le prompt sans prédiction, RAG ou LLM."""
        return PromptCheckResponse(**evaluate_prompt(payload.instruction))

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            return orchestrator.analyze(request)
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("Échec de l'orchestration")
            raise HTTPException(status_code=503, detail=f"Orchestrateur indisponible: {error}") from error

    @app.post("/analyze/stream")
    async def analyze_stream(request: AnalyzeRequest):
        """Flux SSE des étapes réellement exécutées par le graphe LangGraph."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def emit(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"event": "step", "data": event}
            )

        def work() -> None:
            try:
                response = orchestrator.analyze(request, emit=emit)
            except Exception as error:
                logger.exception("Échec de l'orchestration en flux")
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "error", "data": {"detail": f"{type(error).__name__}: {error}"}},
                )
            else:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "result", "data": response.model_dump(mode="json")},
                )

        async def events() -> AsyncIterator[str]:
            task = asyncio.create_task(asyncio.to_thread(work))
            try:
                while True:
                    item = await queue.get()
                    payload = json.dumps(item["data"], ensure_ascii=False)
                    yield f"event: {item['event']}\ndata: {payload}\n\n"
                    if item["event"] in {"result", "error"}:
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
