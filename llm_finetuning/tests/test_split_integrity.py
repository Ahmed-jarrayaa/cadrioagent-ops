"""Contrôle des fuites entre train / validation / test."""

from __future__ import annotations

from collections import defaultdict

from conftest import SPLITS


def test_card_ids_are_disjoint(dataset):
    per_split = {
        split: {record["meta"]["card_id"] for record in dataset[split]} for split in SPLITS
    }
    assert per_split["train"].isdisjoint(per_split["validation"])
    assert per_split["train"].isdisjoint(per_split["test"])
    assert per_split["validation"].isdisjoint(per_split["test"])


def test_no_duplicate_input_output_across_splits(dataset):
    seen: dict[tuple[str, str], str] = {}
    for split in SPLITS:
        for record in dataset[split]:
            key = (record["input"].strip().lower(), record["output"].strip().lower())
            if key in seen:
                assert seen[key] == split, f"Doublon entre {seen[key]} et {split}: {record['id']}"
            seen[key] = split


def test_variants_of_a_card_stay_in_one_split(dataset):
    card_splits = defaultdict(set)
    for split in SPLITS:
        for record in dataset[split]:
            card_splits[record["meta"]["card_id"]].add(split)
    leaks = {card: splits for card, splits in card_splits.items() if len(splits) > 1}
    assert not leaks, f"Fuite de fiche entre splits : {leaks}"


def test_each_card_has_six_variants_in_non_guardrail(dataset):
    counts = defaultdict(int)
    for split in SPLITS:
        for record in dataset[split]:
            if record["meta"]["format"] != "guardrail":
                counts[record["meta"]["card_id"]] += 1
    assert set(counts.values()) == {6}, sorted(set(counts.values()))


def test_guardrails_are_distributed(dataset):
    for split in SPLITS:
        guardrails = [r for r in dataset[split] if r["meta"]["format"] == "guardrail"]
        assert guardrails, split
        assert {r["meta"]["lang"] for r in guardrails} == {"fr", "en"}, split
