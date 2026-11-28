"""Clients des dépendances externes (Qdrant pour le RAG, Ollama pour le LLM).

Les imports lourds (``sentence-transformers``, ``torch``) sont différés : le
service démarre même si l'index ou le modèle ne sont pas encore chargés.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

import httpx


class Retriever(Protocol):
    """Interface minimale attendue par le graphe (injectable en test)."""

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...


class LLM(Protocol):
    """Interface minimale du client LLM (injectable en test)."""

    def generate(self, prompt: str, system: str) -> str: ...

    def health(self) -> dict[str, Any]: ...


class QdrantRetriever:
    """Adaptateur du pipeline ``rag_engine`` vers Qdrant."""

    def __init__(
        self,
        url: str,
        collection: str,
        model_name: str,
        min_score: float = 0.35,
    ) -> None:
        self.url = url
        self.collection = collection
        self.model_name = model_name
        self.min_score = min_score
        self._client = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_client(self):
        with self._lock:
            if self._client is None:
                from rag_engine.index_qdrant import make_client

                self._client = make_client(self.url)
        return self._client

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from rag_engine.embed import load_model

                self._model = load_model(self.model_name)
        return self._model

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        from rag_engine.retrieve import retrieve

        return retrieve(
            query,
            top_k=top_k,
            client=self._ensure_client(),
            model=self._ensure_model(),
            collection=self.collection,
            model_name=self.model_name,
            min_score=self.min_score,
        )

    def health(self) -> dict[str, Any]:
        info = self._ensure_client().get_collection(self.collection)
        return {
            "status": "ok",
            "collection": self.collection,
            "points": int(info.points_count or 0),
            "embedding_model": self.model_name,
            "url": self.url,
        }


class OllamaClient:
    """Client HTTP minimal de l'API Ollama (``/api/generate``)."""

    def __init__(
        self,
        url: str,
        model: str,
        timeout_s: float = 120.0,
        temperature: float = 0.2,
        num_predict: int = 320,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_predict = num_predict

    def generate(self, prompt: str, system: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(f"{self.url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise RuntimeError("Réponse vide du serveur Ollama")
        return text

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{self.url}/api/tags")
            response.raise_for_status()
            names = {item.get("name", "") for item in response.json().get("models", [])}
        available = self.model in names or f"{self.model}:latest" in names
        return {
            "status": "ok" if available else "degraded",
            "model": self.model,
            "model_available": available,
            "url": self.url,
        }
