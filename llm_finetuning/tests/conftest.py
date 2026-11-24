"""Fixtures partagées pour les tests du module LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLM_DIR = REPO_ROOT / "llm_finetuning"
DATASET_DIR = LLM_DIR / "data" / "dataset"
DATA_DIR = LLM_DIR / "data"
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 450, "validation": 70, "test": 70}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def dataset() -> dict[str, list[dict]]:
    return {split: read_jsonl(DATASET_DIR / f"{split}.jsonl") for split in SPLITS}


@pytest.fixture(scope="session")
def sources() -> dict:
    return json.loads((DATA_DIR / "sources.json").read_text(encoding="utf-8"))
