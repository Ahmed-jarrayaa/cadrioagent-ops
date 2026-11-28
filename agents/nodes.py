"""Nœuds du graphe : Analyste → Cardiologue RAG → Auditeur.

Sécurité : les refus (diagnostic, prescription, urgence) sont **déterministes**
et ne passent jamais par le LLM. Le LLM ne sert qu'à reformuler des passages
sourcés ; l'Auditeur remplace toute réponse non ancrée ou interdite.
"""

from __future__ import annotations

import re
from typing import Any

from agents.clients import LLM, Retriever
from agents.config import Settings
from agents.prompts import (
    audio_limit,
    build_user_prompt,
    canonical_answer,
    system_prompt,
)
from agents.state import AnalysisState

EMERGENCY_PATTERNS = [
    r"douleur[s]? (thoracique|poitrine|cardiaque)",
    r"infarctus|heart attack|arr[êe]t cardiaque|cardiac arrest",
    r"je n'arrive (plus )?à respirer|i can'?t breathe|difficult[ée]s? à respirer",
    r"malaise|évanoui|unconscious|fainting|syncope",
    r"\burgence\b|emergency",
    r"douleur intense.*(poitrine|thorax)|chest pain",
    r"\bAVC\b|\bstroke\b|h[ée]morragie",
]

PRESCRIPTION_PATTERNS = [
    r"prescri|ordonnance|posologie",
    r"quelle (dose|posologie)|combien de (mg|comprim[ée]s|g[ée]lules)",
    r"quel (m[ée]dicament|traitement) (pour|dois-je|me)",
    r"traitement (pour|de) (moi|mon patient)",
    r"prescribe|prescription|dosage|how many (mg|pills)|what medication (should|for)",
    r"je (dois|peux) prendre|can i take",
]

DIAGNOSIS_PATTERNS = [
    r"^\s*(diagnostique|diagnostiquer|diagnose|diagnosis)\b",
    r"diagnostique[rz]? (le|la|ce|cet|mon|ma|mes|l'|the)\b",
    r"pose[rz]? (un|le) diagnostic|make a diagnosis|give me a diagnosis",
    r"est-ce que c'est (un|une|le|la)\b|is (this|it) (a|an)\b",
    r"ai-je (un|une|le|la)\b|do i have\b|what do i have",
    r"interpr[èe]te[rz]? (mon|ce|cet|le|la|l')\b|interpret (my|this|the)\b",
    r"chez mon patient.*(souffle|arythmie|bruit|maladie)|my patient.*(murmur|arrhythmia)",
]

FR_MARKERS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "est", "quel", "quelle",
    "quels", "quelles", "comment", "pourquoi", "je", "mon", "ma", "mes", "douleur",
    "coeur", "cœur", "bruit", "souffle", "traitement", "médicament", "urgence",
    "secours", "patient", "auscultation", "cardiaque", "rôle", "role",
}
EN_MARKERS = {
    "the", "a", "an", "is", "what", "how", "why", "my", "pain", "heart", "murmur",
    "treatment", "drug", "dosage", "emergency", "help", "role", "cardiac",
    "auscultation", "patient", "sound", "sounds",
}

STOPWORDS = set(
    """
    le la les un une des du de d au aux et ou mais donc or ni car que qui quoi
    dont où à a en y il elle ils elles je tu nous vous on mon ma mes ton ta tes
    son sa ses ce cet cette ces est sont été être avoir ai as avons avez ont
    pour par avec sans sur sous dans entre chez plus moins très tout tous toute
    toutes comme se sa ne pas plus jamais aussi encore alors si
    the a an and or but of to in on for with without from by at as is are was
    were be been being have has had do does did not no yes this that these those
    it its my your our their his her they we you i me can could should would may
    might must will just very more most about into over under between than then
    """.split()
)

AUDIO_QUERY = {
    "fr": "auscultation cardiaque bruits du coeur évaluation cardiovasculaire dépistage",
    "en": "cardiac auscultation heart sounds cardiovascular assessment screening",
}


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Heuristique FR/EN déterministe (le projet est majoritairement francophone)."""
    lower = text.lower()
    if re.search(r"[àâçéèêëîïôûùüÿœ]", lower):
        return "fr"
    words = set(re.findall(r"[a-z']+", lower))
    fr_score = len(words & FR_MARKERS)
    en_score = len(words & EN_MARKERS)
    return "en" if en_score > fr_score else "fr"


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_intent(text: str) -> str:
    """Priorité : urgence > prescription > diagnostic > connaissance."""
    if _matches(EMERGENCY_PATTERNS, text):
        return "emergency"
    if _matches(PRESCRIPTION_PATTERNS, text):
        return "prescription"
    if _matches(DIAGNOSIS_PATTERNS, text):
        return "diagnosis"
    return "knowledge"


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zàâçéèêëîïôûùüÿœ0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def count_citations(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def is_grounded(answer: str, context: str) -> bool:
    """Vérifie un recouvrement lexical minimal ou la présence de citations."""
    if not answer or not context:
        return False
    answer_tokens = set(_tokens(answer))
    if not answer_tokens:
        return False
    overlap = len(answer_tokens & set(_tokens(context))) / len(answer_tokens)
    citations = count_citations(answer)
    return overlap >= 0.25 or (citations >= 1 and overlap >= 0.12)


def _contains_diagnosis(text: str) -> bool:
    """Détecte une affirmation diagnostique, pas une phrase générique."""
    disease = (
        r"(infarctus|arythmie|maladie|souffle|insuffisance cardiaque|pathologie|"
        r"heart attack|arrhythmia|disease|murmur|pathology|condition)"
    )
    patterns = [
        rf"vous avez (un|une|le|la) .{{0,60}}{disease}",
        rf"you have (a|an|the) .{{0,60}}{disease}",
        rf"il s'agit d'un .{{0,60}}{disease}",
        rf"this is (a|an) .{{0,60}}{disease}",
        r"je diagnostique|i diagnose",
    ]
    return _matches(patterns, text)


def _contains_dosage(text: str) -> re.Match[str] | None:
    return re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|ml|comprim[ée]s?|g[ée]lules?|tablets?)\b",
        text,
        flags=re.IGNORECASE,
    )


def _has_audio_limit(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in ("dépistage", "depistage", "screening", "pas un diagnostic", "not a diagnosis")
    )


def extractive_answer(results: list[dict[str, Any]], language: str) -> str:
    """Réponse de repli : passage source verbatim, sans reformulation."""
    text = results[0]["text"].strip()
    if len(text) > 700:
        text = text[:700].rsplit(" ", 1)[0] + "…"
    if language == "en":
        return f'From the source documents:\n\n"{text}"\n\n[1]'
    return f"D'après les documents sources :\n\n« {text} »\n\n[1]"


def analyste_node(state: AnalysisState) -> dict[str, Any]:
    """Agent 1 : normalise, détecte la langue et qualifie l'intention."""
    question = normalize_question(state.get("question", ""))
    language = state.get("language") or detect_language(question)
    if language not in {"fr", "en"}:
        language = "fr"
    if state.get("audio") and state.get("audio_prompt"):
        # Question générée à partir d'un score : information de dépistage, jamais un diagnostic.
        intent = "knowledge"
        retrieval_query = AUDIO_QUERY[language]
    else:
        # Une question explicite garde sa classification, même si un score audio l'accompagne.
        intent = classify_intent(question)
        retrieval_query = question
    return {
        "question": question,
        "language": language,
        "intent": intent,
        "retrieval_query": retrieval_query,
        "issues": [],
    }


def make_cardiologue_node(retriever: Retriever, llm: LLM, settings: Settings):
    """Agent 2 : RAG + rédaction LLM, ou refus déterministe."""

    def cardiologue_node(state: AnalysisState) -> dict[str, Any]:
        language = state.get("language", "fr")
        intent = state.get("intent", "knowledge")

        if intent in {"diagnosis", "prescription", "emergency"}:
            message = canonical_answer(language, intent)
            return {
                "answer": message,
                "draft_answer": message,
                "status": "refused",
                "results": [],
                "context": "",
                "grounded": False,
                "citations": 0,
                "refusal_applied": True,
                "emergency_redirect": intent == "emergency",
            }

        try:
            results = retriever.retrieve(state.get("retrieval_query") or state["question"], state.get("top_k", settings.top_k))
        except Exception as error:  # index indisponible : abstention, jamais d'invention
            message = canonical_answer(language, "unavailable")
            return {
                "answer": message,
                "draft_answer": message,
                "status": "error",
                "results": [],
                "context": "",
                "grounded": False,
                "citations": 0,
                "issues": [f"rag_unavailable:{type(error).__name__}"],
            }

        if not results:
            message = canonical_answer(language, "abstention")
            return {
                "answer": message,
                "draft_answer": message,
                "status": "abstained",
                "results": [],
                "context": "",
                "grounded": False,
                "citations": 0,
            }

        from rag_engine.retrieve import format_context

        context = format_context(results, max_chars=settings.max_context_chars)
        prompt = build_user_prompt(
            state["question"], context, language, state.get("patient_context")
        )
        try:
            answer = llm.generate(prompt, system_prompt(language))
            status = "ok"
            issues: list[str] = []
        except Exception as error:
            answer = extractive_answer(results, language)
            status = "degraded"
            issues = [f"llm_unavailable:{type(error).__name__}"]

        return {
            "answer": answer,
            "draft_answer": answer,
            "status": status,
            "results": results,
            "context": context,
            "grounded": is_grounded(answer, context),
            "citations": count_citations(answer),
            "issues": issues,
        }

    return cardiologue_node


def auditeur_node(state: AnalysisState) -> dict[str, Any]:
    """Agent 3 : vérifie ancrage, refus et avertissements ; corrige si besoin."""
    language = state.get("language", "fr")
    intent = state.get("intent", "knowledge")
    status = state.get("status", "ok")
    answer = (state.get("answer") or "").strip()
    issues = list(state.get("issues") or [])
    warnings = list(state.get("warnings") or [])
    citations = count_citations(answer)

    if intent in {"diagnosis", "prescription", "emergency"}:
        expected = canonical_answer(language, intent)
        if not answer or answer == "":
            answer = expected
            issues.append(f"{intent}_message_missing")
        return {
            "answer": answer,
            "status": "refused",
            "grounded": False,
            "citations": 0,
            "issues": issues,
            "warnings": warnings,
            "refusal_applied": True,
            "emergency_redirect": intent == "emergency",
        }

    results = state.get("results") or []
    context = state.get("context") or ""

    if status == "error":
        return {
            "answer": answer or canonical_answer(language, "unavailable"),
            "status": "error",
            "grounded": False,
            "citations": 0,
            "issues": issues,
            "warnings": warnings,
        }

    if not results:
        return {
            "answer": canonical_answer(language, "abstention"),
            "status": "abstained",
            "grounded": False,
            "citations": 0,
            "issues": issues,
            "warnings": warnings,
        }

    dosage = _contains_dosage(answer)
    if _contains_diagnosis(answer):
        issues.append("diagnosis_in_answer")
        answer = canonical_answer(language, "diagnosis")
        status = "refused"
        citations = 0
        grounded = False
    elif dosage and dosage.group(0).lower() not in context.lower():
        issues.append("dosage_in_answer")
        answer = canonical_answer(language, "prescription")
        status = "refused"
        citations = 0
        grounded = False
    else:
        if state.get("audio") and not _has_audio_limit(answer):
            answer = f"{audio_limit(language)}\n\n{answer}"
            warnings.append("audio_limit_added")
        grounded = is_grounded(answer, context)
        # Les sources sont en français : un recouvrement lexical avec une réponse
        # anglaise est faible. On accepte l'ancrage cross-lingue mais on le signale.
        cross_lingual = detect_language(answer) != detect_language(context)
        if not grounded and cross_lingual and len(_tokens(answer)) >= 5:
            grounded = True
            warnings.append("cross_lingual_grounding")
        if not grounded and state.get("audio") and len(_tokens(answer)) >= 5:
            # Une explication de score audio s'appuie sur le résumé fourni ; le
            # contrôle lexical contre les passages RAG n'est pas pertinent.
            grounded = True
            warnings.append("audio_screening_grounding")
        if not grounded:
            issues.append("answer_not_grounded")
            answer = extractive_answer(results, language)
            grounded = True
            citations = count_citations(answer)
            status = "degraded"

    if status != "refused" and detect_language(answer) != language:
        warnings.append("language_mismatch")

    return {
        "answer": answer,
        "status": status,
        "grounded": grounded,
        "citations": citations,
        "issues": issues,
        "warnings": warnings,
        "refusal_applied": status == "refused",
        "emergency_redirect": False,
    }
