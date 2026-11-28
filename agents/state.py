"""État partagé du graphe LangGraph (3 agents linéaires)."""

from __future__ import annotations

from typing import Any, TypedDict


class AnalysisState(TypedDict, total=False):
    """Données transmises entre Analyste → Cardiologue RAG → Auditeur."""

    question: str
    retrieval_query: str
    language: str
    intent: str
    top_k: int
    patient_context: str | None
    audio: dict[str, Any] | None
    audio_prompt: bool
    mode: str
    workflow: dict[str, Any] | None
    workflow_summary: str | None
    factual_report: str | None

    results: list[dict[str, Any]]
    context: str
    draft_answer: str
    answer: str
    status: str
    grounded: bool
    citations: int
    issues: list[str]
    warnings: list[str]
    refusal_applied: bool
    emergency_redirect: bool
