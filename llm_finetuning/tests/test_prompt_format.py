"""Validation du format des prompts et de la présence des garde-fous."""

from __future__ import annotations

from conftest import SPLITS
from llm_finetuning.src.data import build_messages, build_prompt_messages

SAFETY_MARKERS = {
    "fr": (
        "diagnostic",
        "médecin",
        "urgences",
        "secours",
        "consultez",
        "professionnel de santé",
        "je ne peux pas",
        "avis médical",
        "évaluation médicale",
        "spécialiste",
        "clinicien",
        "équipe médicale",
        "collégiale",
    ),
    "en": (
        "diagnos",
        "physician",
        "emergency",
        "doctor",
        "consult",
        "healthcare professional",
        "i cannot",
        "medical advice",
        "medical assessment",
        "specialist",
        "clinician",
        "medical team",
        "multidisciplinary",
    ),
}

# Sujets de cadre : réponse déontologique attendue (pas un refus).
DEONTOLOGY_TOPICS = {
    "sources",
    "status",
    "statut",
    "privacy",
    "confidentialite",
    "confidentialité",
    "role",
    "rôle",
    "limitations",
    "limites",
    "education",
    "éducation",
}


def test_build_messages_roles(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            messages = build_messages(record)
            assert [message["role"] for message in messages] == ["system", "user", "assistant"]
            for message in messages:
                assert message["content"].strip()


def test_generation_prompt_has_no_answer(dataset):
    record = dataset["train"][0]
    messages = build_prompt_messages(record)
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"


def test_guardrail_count(dataset):
    total = sum(
        1
        for split in SPLITS
        for record in dataset[split]
        if record["meta"]["format"] == "guardrail"
    )
    assert total == 50


def test_guardrail_answers_redirect(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            if record["meta"]["format"] != "guardrail":
                continue
            if record["meta"].get("topic") in DEONTOLOGY_TOPICS:
                continue
            output = record["output"].lower()
            markers = SAFETY_MARKERS[record["meta"]["lang"]]
            assert any(marker in output for marker in markers), record["id"]


def test_each_split_contains_both_languages(dataset):
    for split in SPLITS:
        languages = {record["meta"]["lang"] for record in dataset[split]}
        assert languages == {"fr", "en"}


def test_no_diagnosis_instruction(dataset):
    # Seules les consignes système sont contrôlées : les entrées utilisateur peuvent
    # légitimement demander un diagnostic, l'assistant doit alors refuser.
    forbidden = ("tu dois poser un diagnostic", "you must diagnose", "confirme le diagnostic")
    for split in SPLITS:
        for record in dataset[split]:
            text = record["instruction"].lower()
            assert not any(phrase in text for phrase in forbidden), record["id"]
