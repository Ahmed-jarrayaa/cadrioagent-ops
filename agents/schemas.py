"""Schémas Pydantic du service agent-orchestrator.

Le contrat d'entrée accepte soit une question libre, soit le résultat audio
envoyé par ``/analyze_pcg`` (payload historique de l'audio-engine). Le contrat
de sortie est strict : toute réponse expose ses sources, son audit et ses
garde-fous.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["fr", "en"]
Intent = Literal["knowledge", "report", "diagnosis", "prescription", "emergency", "out_of_scope"]
Status = Literal["ok", "refused", "abstained", "degraded", "error"]


class AnalyzeRequest(BaseModel):
    """Requête d'analyse : question libre ou résultat audio (legacy)."""

    model_config = ConfigDict(extra="allow")

    question: str | None = Field(default=None, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    language: Literal["fr", "en", "auto"] = "auto"
    patient_context: str | None = Field(default=None, max_length=1000)

    # Mode « rapport » : résultats réels du workflow 5 modèles + instruction.
    mode: Literal["chat", "report"] = "chat"
    workflow: dict[str, Any] | None = None

    # Champs du payload audio-engine (POST /predict) transmis par /analyze_pcg.
    filename: str | None = None
    model: str | None = None
    duration_s: float | None = None
    n_segments: int | None = None
    prob_abnormal: float | None = Field(default=None, ge=0, le=1)
    prob_abnormal_max: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    prediction: str | None = None
    segment_probs: list[float] | None = None
    features: dict[str, float] | None = None

    @model_validator(mode="after")
    def _require_question_or_audio(self):
        if self.question is not None and not self.question.strip():
            raise ValueError("'question' ne peut pas être vide.")
        has_audio = self.prediction is not None or self.prob_abnormal is not None
        if not self.question and not has_audio:
            raise ValueError("Fournir 'question' ou un résultat audio (prediction / prob_abnormal).")
        return self


class AudioSummary(BaseModel):
    """Résumé du score audio, sans interprétation diagnostique."""

    prediction: str
    prob_abnormal: float | None = None
    threshold: float | None = None
    model: str | None = None
    filename: str | None = None


class SourceRef(BaseModel):
    """Provenance d'un passage utilisé pour ancrer la réponse."""

    source_id: str
    source: str
    page: int
    score: float
    license: str
    chunk_id: str | None = None


class AuditReport(BaseModel):
    """Contrôle de l'agent Auditeur."""

    passed: bool
    grounded: bool
    citations: int = 0
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    """Garde-fous effectivement appliqués."""

    refusal_applied: bool
    emergency_redirect: bool
    diagnosis_blocked: bool
    prescription_blocked: bool


class AnalyzeResponse(BaseModel):
    """Réponse structurée, validée par Pydantic avant envoi."""

    status: Status
    language: Language
    intent: Intent
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    audit: AuditReport
    guardrails: GuardrailReport
    audio: AudioSummary | None = None
    model: str
    latency_ms: float
    disclaimer: str
