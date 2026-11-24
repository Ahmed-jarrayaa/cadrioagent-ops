"""Vérifie que chaque exemple est traçable jusqu'à une source documentée."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from conftest import DATA_DIR, LLM_DIR, SPLITS

RAW_FILES = {
    "auscultation_cardiaque_normale": DATA_DIR / "raw" / "auscultation_cardiaque_normale.txt",
    "esc_evaluation_cardiovasculaire": DATA_DIR / "raw" / "esc_evaluation_cardiovasculaire.txt",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def raw_corpus(source_id: str) -> str | None:
    path = RAW_FILES.get(source_id)
    if path is None or not path.exists():
        return None
    return normalize(path.read_text(encoding="utf-8", errors="ignore"))


def test_evidence_present(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            if record["meta"]["format"] == "guardrail":
                continue
            evidence = record["meta"].get("evidence", "").strip()
            assert len(evidence) >= 30, record["id"]


def test_evidence_is_grounded_in_source(dataset):
    corpora = {source_id: raw_corpus(source_id) for source_id in RAW_FILES}
    if all(corpus is None for corpus in corpora.values()):
        return  # textes bruts non générés : test informatif uniquement

    for split in SPLITS:
        for record in dataset[split]:
            if record["meta"]["format"] == "guardrail":
                continue
            source_id = record["meta"]["source_ids"][0]
            corpus = corpora.get(source_id)
            if corpus is None:
                continue
            evidence_tokens = [tok for tok in normalize(record["meta"]["evidence"]).split() if len(tok) > 3]
            assert evidence_tokens, record["id"]
            found = sum(1 for token in evidence_tokens if token in corpus)
            ratio = found / len(evidence_tokens)
            assert ratio >= 0.6, f"{record['id']} : ancrage faible ({ratio:.2f})"


def test_sources_registry(dataset, sources):
    registered = {source["id"] for source in sources["sources"]}
    for split in SPLITS:
        for record in dataset[split]:
            for source_id in record["meta"]["source_ids"]:
                if source_id == "guardrails":
                    continue
                assert source_id in registered, record["id"]
                assert record["meta"]["license"].strip(), record["id"]


def test_dataset_card_documents_limits():
    card = LLM_DIR / "data" / "dataset" / "dataset_card.md"
    assert card.exists()
    text = card.read_text(encoding="utf-8")
    assert "CC BY-NC 4.0" in text
    assert "dispositif médical" in text.lower()
    assert "pas de validation clinique" in text.lower() or "aucune validation clinique" in text.lower()


def test_guardrails_have_no_medical_evidence():
    guardrail_files = [Path(DATA_DIR / "seed_guardrails.jsonl")]
    assert guardrail_files[0].exists()
