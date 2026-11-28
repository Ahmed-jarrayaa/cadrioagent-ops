"""Configuration du service, lue depuis l'environnement (docker-compose)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Paramètres de l'orchestrateur et des dépendances."""

    llm_url: str = "http://127.0.0.1:11434"
    llm_model: str = "cardio-llm"
    llm_timeout_s: float = 120.0
    llm_temperature: float = 0.2
    llm_num_predict: int = 320

    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "cardio_docs"
    embedding_model: str = "BAAI/bge-m3"
    min_score: float = 0.35
    top_k: int = 4
    max_context_chars: int = 4000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_url=os.getenv("LLM_URL", cls.llm_url),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_timeout_s=_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            llm_temperature=_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_num_predict=_int("LLM_NUM_PREDICT", cls.llm_num_predict),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
            collection=os.getenv("QDRANT_COLLECTION", cls.collection),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            min_score=_float("RAG_MIN_SCORE", cls.min_score),
            top_k=_int("RAG_TOP_K", cls.top_k),
            max_context_chars=_int("RAG_MAX_CONTEXT_CHARS", cls.max_context_chars),
        )
