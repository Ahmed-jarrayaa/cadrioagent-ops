"""Validation du schéma et des volumes du dataset JSONL."""

from __future__ import annotations

from conftest import EXPECTED_COUNTS, SPLITS

REQUIRED_FIELDS = ("instruction", "input", "output")
REQUIRED_META = ("lang", "format", "split", "source_ids", "card_id", "topic", "license")
ALLOWED_FORMATS = {"qa_fr", "qa_en", "patient_fr", "patient_en", "reverse_fr", "reverse_en", "guardrail"}


def test_volumes_match_target(dataset):
    for split in SPLITS:
        assert len(dataset[split]) == EXPECTED_COUNTS[split], split


def test_total_volume(dataset):
    assert sum(len(items) for items in dataset.values()) == 590


def test_required_fields(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            for field in REQUIRED_FIELDS:
                assert isinstance(record.get(field), str)
                assert record[field].strip(), f"{record.get('id')} : {field} vide"


def test_metadata_fields(dataset, sources):
    known_sources = {source["id"] for source in sources["sources"]} | {"guardrails"}
    for split in SPLITS:
        for record in dataset[split]:
            meta = record.get("meta")
            assert isinstance(meta, dict), record.get("id")
            for field in REQUIRED_META:
                assert field in meta, f"{record['id']} : meta.{field} absent"
            assert meta["split"] == split
            assert meta["lang"] in {"fr", "en"}
            assert meta["format"] in ALLOWED_FORMATS
            assert isinstance(meta["source_ids"], list) and meta["source_ids"]
            assert set(meta["source_ids"]).issubset(known_sources)


def test_unique_ids(dataset):
    ids = [record["id"] for split in SPLITS for record in dataset[split]]
    assert len(ids) == len(set(ids))


def test_reasonable_lengths(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            assert len(record["input"]) < 4000, record["id"]
            assert 5 <= len(record["output"]) < 4000, record["id"]


def test_language_balance_per_split(dataset):
    for split in SPLITS:
        langs = {record["meta"]["lang"] for record in dataset[split]}
        assert langs == {"fr", "en"}, split


def test_no_empty_output(dataset):
    for split in SPLITS:
        for record in dataset[split]:
            assert record["output"].strip() not in {"-", "N/A", "n/a"}
