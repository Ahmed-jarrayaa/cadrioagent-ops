"""Tests du graphe multi-agents et des garde-fous Pydantic.

Aucun appel réseau : le retriever et le LLM sont des doubles de test.
Les refus doivent être déterministes et ne jamais dépendre du LLM.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agents.config import Settings
from agents.main import create_app
from agents.nodes import classify_intent, detect_language
from agents.prompts import (
    ABSTENTION_EN,
    EMERGENCY_EN,
    EMERGENCY_FR,
    PRESCRIPTION_EN,
    PRESCRIPTION_FR,
    REFUSAL_EN,
    REFUSAL_FR,
    SERVICE_UNAVAILABLE_FR,
)
from agents.schemas import AnalyzeRequest
from agents.service import Orchestrator

B1_RESULT = {
    "score": 0.631,
    "text": (
        "Le B1 est intense, grave, sourd et prolongé, contemporain de la fermeture "
        "des valves auriculo-ventriculaires."
    ),
    "source": "Auscultation-cardiaque-normale-4-4.pdf",
    "source_id": "auscultation_cardiaque_normale",
    "page": 5,
    "chunk_id": "chunk_b1",
    "license": "usage académique interne",
    "disclaimer": "Usage pédagogique. Pas un avis médical.",
}

GROUNDED_ANSWER = (
    "Le B1 est un bruit intense, grave et sourd [1]. Il correspond à la fermeture "
    "des valves auriculo-ventriculaires [1]. Information pédagogique, pas un avis médical."
)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self.results[:top_k]

    def health(self):
        return {"status": "ok", "collection": "test", "points": len(self.results)}


class FakeLLM:
    def __init__(self, reply=GROUNDED_ANSWER):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, system):
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply

    def health(self):
        return {"status": "ok", "model": "fake", "model_available": True}


class BrokenRetriever:
    """Qdrant indisponible : retrieve et health échouent."""

    def retrieve(self, query, top_k):
        raise ConnectionError("Qdrant arrêté")

    def health(self):
        raise ConnectionError("Qdrant arrêté")


class FailingLLM:
    """LLM indisponible : la réponse doit basculer en extraction source."""

    def generate(self, prompt, system):
        raise TimeoutError("Ollama indisponible")

    def health(self):
        return {"status": "ok", "model": "fake", "model_available": True}


def _orchestrator(retriever=None, llm=None):
    return Orchestrator(
        retriever=retriever or FakeRetriever([B1_RESULT]),
        llm=llm or FakeLLM(),
        settings=Settings(llm_url="http://test", qdrant_url="http://test"),
    )


# --- Analyste ----------------------------------------------------------------


def test_detection_langue():
    assert detect_language("Quel est le rôle de l'auscultation cardiaque ?") == "fr"
    assert detect_language("What is the role of cardiac auscultation?") == "en"


def test_classification_intentions():
    assert classify_intent("Qu'est-ce que le premier bruit B1 ?") == "knowledge"
    assert classify_intent("Comment diagnostiquer une arythmie ?") == "knowledge"
    assert classify_intent("Diagnostique le souffle de mon patient.") == "diagnosis"
    assert classify_intent("Quelle posologie pour ce médicament ?") == "prescription"
    assert classify_intent("J'ai une douleur thoracique intense.") == "emergency"


# --- Parcours nominal --------------------------------------------------------


def test_parcours_connaissance_avec_sources():
    llm = FakeLLM()
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="Qu'est-ce que le premier bruit B1 ?")
    )
    assert response.status == "ok"
    assert response.intent == "knowledge"
    assert response.language == "fr"
    assert len(response.sources) == 1
    assert response.sources[0].source_id == "auscultation_cardiaque_normale"
    assert response.audit.passed is True
    assert response.audit.grounded is True
    assert response.audit.citations >= 1
    assert "B1" in response.answer
    assert "pas un avis médical" in response.disclaimer
    assert len(llm.calls) == 1
    assert "PASSAGES SOURCES" in llm.calls[0]["prompt"]


def test_reponse_anglaise():
    response = _orchestrator().analyze(
        AnalyzeRequest(question="What is the role of cardiac auscultation?")
    )
    assert response.language == "en"
    assert response.disclaimer.startswith("Educational information")


# --- Garde-fous déterministes ------------------------------------------------


def test_refus_diagnostic_sans_appeler_le_llm():
    llm = FakeLLM("Vous avez un infarctus. Prenez 5 mg de bisoprolol.")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="Diagnostique le souffle que j'entends chez mon patient.")
    )
    assert response.status == "refused"
    assert response.intent == "diagnosis"
    assert response.answer == REFUSAL_FR
    assert response.guardrails.refusal_applied is True
    assert response.guardrails.diagnosis_blocked is True
    assert llm.calls == []


def test_question_explicite_garde_sa_classification_avec_audio():
    llm = FakeLLM()
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(
            question="Diagnostique ce souffle.",
            prediction="abnormal",
            prob_abnormal=0.9,
        )
    )
    assert response.intent == "diagnosis"
    assert response.status == "refused"
    assert llm.calls == []


def test_urgence_redirige_vers_les_secours():
    llm = FakeLLM("Ce n'est rien, reposez-vous.")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="J'ai une douleur thoracique intense, est-ce un infarctus ?")
    )
    assert response.status == "refused"
    assert response.intent == "emergency"
    assert response.answer == EMERGENCY_FR
    assert response.guardrails.emergency_redirect is True
    assert "secours" in response.answer.lower()
    assert llm.calls == []


def test_prescription_refusee():
    llm = FakeLLM("Prenez 5 mg de bisoprolol matin et soir.")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="Quelle posologie de bisoprolol pour mon patient ?")
    )
    assert response.status == "refused"
    assert response.intent == "prescription"
    assert response.answer == PRESCRIPTION_FR
    assert response.guardrails.prescription_blocked is True
    assert llm.calls == []


# --- Abstention et audit -----------------------------------------------------


def test_absence_de_contexte_abstention_sans_llm():
    llm = FakeLLM()
    response = _orchestrator(retriever=FakeRetriever([]), llm=llm).analyze(
        AnalyzeRequest(question="Quel est le prix d'un billet de train ?")
    )
    assert response.status == "abstained"
    assert response.sources == []
    assert "documents sources" in response.answer
    assert response.audit.passed is True
    assert llm.calls == []


def test_auditeur_remplace_reponse_non_fondee():
    llm = FakeLLM("La lune est en fromage.")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="Qu'est-ce que le premier bruit B1 ?")
    )
    assert "answer_not_grounded" in response.audit.issues
    assert response.audit.passed is False
    assert response.status == "degraded"
    assert "documents sources" in response.answer
    assert "sourd" in response.answer
    assert len(llm.calls) == 1


def test_auditeur_bloque_un_dosage_non_source():
    llm = FakeLLM("Prenez 5 mg de bisoprolol matin et soir [1].")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="Qu'est-ce que le premier bruit B1 ?")
    )
    assert "dosage_in_answer" in response.audit.issues
    assert response.status == "refused"
    assert response.guardrails.prescription_blocked is True
    assert "5 mg" not in response.answer


# --- Résultat audio ----------------------------------------------------------


def test_resultat_audio_reste_un_depistage():
    llm = FakeLLM(
        "Le B1 est un bruit intense, grave et sourd [1]. "
        "Ce test de dépistage ne remplace pas un examen clinique [1]."
    )
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(prediction="abnormal", prob_abnormal=0.82, threshold=0.5, model="test")
    )
    assert response.intent == "knowledge"
    assert response.audio is not None
    assert response.audio.prediction == "abnormal"
    assert "dépistage" in response.answer.lower()
    assert response.audit.passed is True
    assert "dépistage audio" in response.question.lower()
    assert "dépistage" in llm.calls[0]["prompt"].lower()


# --- API FastAPI -------------------------------------------------------------


def test_api_analyze_et_health():
    app = create_app(
        retriever=FakeRetriever([B1_RESULT]),
        llm=FakeLLM(),
        settings=Settings(llm_url="http://test", qdrant_url="http://test"),
    )
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/analyze", json={"question": "Qu'est-ce que le premier bruit B1 ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sources"][0]["page"] == 5
    assert body["audit"]["passed"] is True


def test_api_payload_audio_et_validation():
    app = create_app(
        retriever=FakeRetriever([B1_RESULT]),
        llm=FakeLLM(),
        settings=Settings(llm_url="http://test", qdrant_url="http://test"),
    )
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={"prediction": "normal", "prob_abnormal": 0.12, "threshold": 0.5, "model": "onnx"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["audio"]["prediction"] == "normal"
    assert body["intent"] == "knowledge"

    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze", json={"question": "   "}).status_code == 422
    assert client.post("/analyze", json={"question": "B1 ?", "top_k": 99}).status_code == 422


def test_validation_pydantic_champs():
    with pytest.raises(ValidationError):
        AnalyzeRequest(question="   ")
    with pytest.raises(ValidationError):
        AnalyzeRequest(question="ok", prob_abnormal=1.5)


# --- Refus et abstention en anglais ------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_message"),
    [
        ("Please diagnose this murmur for me.", REFUSAL_EN),
        ("What dosage should I take?", PRESCRIPTION_EN),
        ("I have severe chest pain, is it a heart attack?", EMERGENCY_EN),
    ],
)
def test_refus_deterministe_en_anglais(question, expected_message):
    llm = FakeLLM("You have a heart attack. Take 5 mg of bisoprolol.")
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question=question, language="en")
    )
    assert response.status == "refused"
    assert response.language == "en"
    assert response.answer == expected_message
    assert response.disclaimer.startswith("Educational information")
    assert llm.calls == []


def test_abstention_en_anglais():
    response = _orchestrator(retriever=FakeRetriever([])).analyze(
        AnalyzeRequest(question="What is the price of a train ticket?", language="en")
    )
    assert response.status == "abstained"
    assert response.answer == ABSTENTION_EN
    assert response.sources == []


# --- Dégradation contrôlée ----------------------------------------------------


def test_qdrant_indisponible_sans_invention():
    llm = FakeLLM()
    response = _orchestrator(retriever=BrokenRetriever(), llm=llm).analyze(
        AnalyzeRequest(question="Qu'est-ce que le premier bruit B1 ?")
    )
    assert response.status == "error"
    assert response.answer == SERVICE_UNAVAILABLE_FR
    assert response.sources == []
    assert any(issue.startswith("rag_unavailable:") for issue in response.audit.issues)
    assert response.audit.passed is False
    assert llm.calls == []


def test_llm_indisponible_repli_extractif():
    response = _orchestrator(llm=FailingLLM()).analyze(
        AnalyzeRequest(question="Qu'est-ce que le premier bruit B1 ?")
    )
    assert response.status == "degraded"
    assert any(issue.startswith("llm_unavailable:") for issue in response.audit.issues)
    assert "documents sources" in response.answer
    assert "sourd" in response.answer  # passage source verbatim
    assert response.sources[0].source_id == "auscultation_cardiaque_normale"


# --- Avertissements de l'auditeur --------------------------------------------


def test_avertissement_ancrage_cross_lingue():
    llm = FakeLLM(
        "The first heart sound is intense, low and dull [1]. "
        "It is educational information, not medical advice."
    )
    response = _orchestrator(llm=llm).analyze(
        AnalyzeRequest(question="What is the first heart sound?", language="en")
    )
    assert response.status == "ok"
    assert "cross_lingual_grounding" in response.audit.warnings
    assert response.audit.grounded is True
    assert response.audit.passed is True


def test_avertissement_langue_incoherente():
    # Le 1.5B répond parfois en français à une question anglaise : l'auditeur le signale.
    response = _orchestrator(llm=FakeLLM()).analyze(
        AnalyzeRequest(question="What is the first heart sound?", language="en")
    )
    assert response.language == "en"
    assert "language_mismatch" in response.audit.warnings
    assert response.audit.passed is True


def test_health_degrade_quand_une_dependance_est_en_panne():
    app = create_app(
        retriever=BrokenRetriever(),
        llm=FakeLLM(),
        settings=Settings(llm_url="http://test", qdrant_url="http://test"),
    )
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["services"]["qdrant"]["status"] == "down"
    assert body["services"]["llm"]["status"] == "ok"
    assert body["model"] == "cardio-llm"


def test_root_orchestrateur():
    app = create_app(
        retriever=FakeRetriever([B1_RESULT]),
        llm=FakeLLM(),
        settings=Settings(llm_url="http://test", qdrant_url="http://test"),
    )
    body = TestClient(app).get("/").json()
    assert body["service"] == "agent-orchestrator"
    assert body["graph"] == ["analyste", "cardiologue_rag", "auditeur"]
