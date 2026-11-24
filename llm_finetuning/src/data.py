"""Préparation des données d'instruction (JSONL) pour QLoRA.

Chaque exemple contient ``instruction``, ``input``, ``output`` et un bloc ``meta``.
Le format d'entraînement suit le template de chat du modèle (ChatML pour Qwen) :

    system      <- instruction (rôle et consigne)
    user        <- input (question / contexte)
    assistant   <- output (réponse)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = ("instruction", "input", "output")


class DatasetError(ValueError):
    """Exemple de dataset invalide."""


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset introuvable : {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{path.name}:{line_number}: JSON invalide ({error})") from error
            validate_record(record, f"{path.name}:{line_number}")
            records.append(record)
    if not records:
        raise DatasetError(f"Dataset vide : {path}")
    return records


def validate_record(record: dict[str, Any], location: str = "<record>") -> None:
    if not isinstance(record, dict):
        raise DatasetError(f"{location}: l'exemple doit être un objet JSON.")
    missing = [field for field in REQUIRED_FIELDS if not str(record.get(field, "")).strip()]
    if missing:
        raise DatasetError(f"{location}: champs manquants ou vides {missing}")
    if "meta" not in record or not isinstance(record["meta"], dict):
        raise DatasetError(f"{location}: bloc meta absent ou invalide.")
    meta = record["meta"]
    for field in ("lang", "format", "split", "source_ids"):
        if field not in meta:
            raise DatasetError(f"{location}: meta.{field} manquant.")
    if meta["lang"] not in {"fr", "en"}:
        raise DatasetError(f"{location}: langue invalide {meta['lang']!r}.")
    if not isinstance(meta["source_ids"], list) or not meta["source_ids"]:
        raise DatasetError(f"{location}: meta.source_ids doit être une liste non vide.")


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Construit les messages ChatML d'un exemple."""
    return [
        {"role": "system", "content": str(record["instruction"]).strip()},
        {"role": "user", "content": str(record["input"]).strip()},
        {"role": "assistant", "content": str(record["output"]).strip()},
    ]


def build_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Messages sans la réponse (pour l'inférence et l'évaluation)."""
    return build_messages(record)[:2]


def format_for_training(record: dict[str, Any], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        build_messages(record), tokenize=False, add_generation_prompt=False
    )


def format_for_generation(record: dict[str, Any], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        build_prompt_messages(record), tokenize=False, add_generation_prompt=True
    )


def load_text_dataset(path: str | Path, tokenizer) -> Any:
    """Charge un JSONL et renvoie un ``datasets.Dataset`` avec le seul champ ``text``.

    Les colonnes de traçabilité (``id``, ``lang``) sont volontairement exclues :
    le collator de TRL attend uniquement le texte à tokeniser, et des colonnes
    supplémentaires provoquent des erreurs de padding.
    """
    from datasets import Dataset

    records = load_records(path)
    texts = [format_for_training(record, tokenizer) for record in records]
    return Dataset.from_dict({"text": texts})


def iter_records(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from load_records(path)


def split_stats(paths: dict[str, str | Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, path in paths.items():
        records = load_records(path)
        stats[split] = {
            "total": len(records),
            "by_lang": dict(Counter(record["meta"]["lang"] for record in records)),
            "by_format": dict(Counter(record["meta"]["format"] for record in records)),
        }
    return stats


def fingerprint(paths: Iterable[str | Path]) -> dict[str, str]:
    """Empreinte SHA-256 des fichiers de dataset (traçabilité MLflow)."""
    result: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[path.name] = digest
    return result
