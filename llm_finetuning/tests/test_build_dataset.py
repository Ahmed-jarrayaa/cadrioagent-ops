"""Tests du générateur de dataset (déterministe, sans fuite)."""

from __future__ import annotations

from collections import Counter

from llm_finetuning.data import build_dataset as builder


def test_build_is_deterministic():
    sources = builder.load_sources()
    first = builder.build(sources)
    second = builder.build(sources)
    for split in builder.SPLITS:
        assert [r["id"] for r in first[split]] == [r["id"] for r in second[split]]


def test_build_volumes():
    sources = builder.load_sources()
    records = builder.build(sources)
    assert {split: len(items) for split, items in records.items()} == {
        "train": 450,
        "validation": 70,
        "test": 70,
    }


def test_build_no_duplicate_ids():
    records = builder.build(builder.load_sources())
    ids = [record["id"] for split in builder.SPLITS for record in records[split]]
    assert len(ids) == len(set(ids))


def test_build_six_variants_per_card():
    records = builder.build(builder.load_sources())
    counts = Counter(
        record["meta"]["card_id"]
        for split in builder.SPLITS
        for record in records[split]
        if record["meta"]["format"] != "guardrail"
    )
    assert len(counts) == 90
    assert set(counts.values()) == {6}


def test_guardrail_distribution_is_bilingual():
    records = builder.build(builder.load_sources())
    for split in builder.SPLITS:
        guardrails = [r for r in records[split] if r["meta"]["format"] == "guardrail"]
        assert {r["meta"]["lang"] for r in guardrails} == {"fr", "en"}
