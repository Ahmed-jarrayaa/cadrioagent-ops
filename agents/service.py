"""Service d'orchestration : requête Pydantic → graphe LangGraph → réponse auditée."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from agents.clients import LLM, Retriever
from agents.config import Settings
from agents.graph import build_graph
from agents.nodes import detect_language
from agents.prompts import DISCLAIMER_EN, DISCLAIMER_FR
from agents.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AudioSummary,
    AuditReport,
    GuardrailReport,
    SourceRef,
)


def build_audio_question(audio: AudioSummary, language: str) -> str:
    """Transforme un score audio en question pédagogique (jamais un diagnostic)."""
    probability = (
        f"{audio.prob_abnormal:.2f}" if audio.prob_abnormal is not None else "non fournie"
    )
    threshold = f"{audio.threshold:.2f}" if audio.threshold is not None else "non fourni"
    model = audio.model or "audio-engine"
    if language == "en":
        return (
            f"Audio screening result: model '{model}' classified the signal as "
            f"'{audio.prediction}' with an abnormal probability of {probability} "
            f"(threshold {threshold}). Explain this result to the patient in simple terms, "
            "without making a diagnosis, and state the limits of a screening test."
        )
    return (
        f"Résultat de dépistage audio : le modèle « {model} » classe le signal comme "
        f"« {audio.prediction} » avec une probabilité d'anomalie de {probability} "
        f"(seuil {threshold}). Explique ce résultat au patient en termes simples, "
        "sans poser de diagnostic, et rappelle les limites d'un test de dépistage."
    )


def audio_from_request(request: AnalyzeRequest) -> AudioSummary | None:
    """Construit le résumé audio si le payload contient un score."""
    if request.prediction is None and request.prob_abnormal is None:
        return None
    threshold = request.threshold if request.threshold is not None else 0.5
    prediction = request.prediction
    if prediction is None:
        prediction = "abnormal" if (request.prob_abnormal or 0.0) >= threshold else "normal"
    return AudioSummary(
        prediction=prediction,
        prob_abnormal=request.prob_abnormal,
        threshold=request.threshold,
        model=request.model,
        filename=request.filename,
    )


@dataclass
class Orchestrator:
    """Assemble les dépendances et exécute le graphe à trois agents."""

    retriever: Retriever
    llm: LLM
    settings: Settings
    _graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._graph = build_graph(self.retriever, self.llm, self.settings)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        started = perf_counter()
        audio = audio_from_request(request)

        if request.question:
            question = request.question.strip()
            language = request.language if request.language in {"fr", "en"} else detect_language(question)
        else:
            language = "en" if request.language == "en" else "fr"
            assert audio is not None
            question = build_audio_question(audio, language)

        initial_state = {
            "question": question,
            "language": language,
            "top_k": request.top_k,
            "patient_context": request.patient_context,
            "audio": audio.model_dump() if audio else None,
            "audio_prompt": bool(audio and not request.question),
        }
        final = self._graph.invoke(initial_state)

        resolved_language = final.get("language") or language
        sources = [
            SourceRef(
                source_id=str(item.get("source_id") or "unknown"),
                source=str(item.get("source") or ""),
                page=int(item.get("page") or 0),
                score=float(item.get("score") or 0.0),
                license=str(item.get("license") or ""),
                chunk_id=item.get("chunk_id"),
            )
            for item in final.get("results", [])
        ]
        issues = list(final.get("issues") or [])
        status = final.get("status", "error")
        intent = final.get("intent", "knowledge")

        audit = AuditReport(
            passed=not issues,
            grounded=bool(final.get("grounded", False)),
            citations=int(final.get("citations") or 0),
            issues=issues,
            warnings=list(final.get("warnings") or []),
        )
        guardrails = GuardrailReport(
            refusal_applied=bool(final.get("refusal_applied", status == "refused")),
            emergency_redirect=bool(final.get("emergency_redirect", False)),
            diagnosis_blocked=(intent != "diagnosis") or status == "refused",
            prescription_blocked=(intent != "prescription") or status == "refused",
        )
        return AnalyzeResponse(
            status=status,
            language="en" if resolved_language == "en" else "fr",
            intent=intent,
            question=question,
            answer=final.get("answer", ""),
            sources=sources,
            audit=audit,
            guardrails=guardrails,
            audio=audio,
            model=self.settings.llm_model,
            latency_ms=round((perf_counter() - started) * 1000, 1),
            disclaimer=DISCLAIMER_EN if resolved_language == "en" else DISCLAIMER_FR,
        )
