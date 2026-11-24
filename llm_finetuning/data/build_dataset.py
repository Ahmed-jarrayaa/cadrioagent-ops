"""Construit le dataset JSONL (instruction, input, output) du fine-tuning LLM.

Le dataset est dérivé de fiches de faits **ancrées** dans deux documents fournis
(voir ``sources.json``). Le script ne génère aucune donnée médicale nouvelle : il
produit plusieurs tâches pédagogiques à partir de chaque fiche (question/réponse
en français et en anglais, explication patient, question inverse) et y ajoute un
jeu de consignes de sécurité (guardrails).

Usage :
    python llm_finetuning/data/build_dataset.py
    python llm_finetuning/data/build_dataset.py --check

Sorties :
    llm_finetuning/data/dataset/train.jsonl
    llm_finetuning/data/dataset/validation.jsonl
    llm_finetuning/data/dataset/test.jsonl
    llm_finetuning/data/dataset/split_manifest.json
    llm_finetuning/data/dataset/dataset_stats.json
    llm_finetuning/data/dataset/dataset_card.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DATASET_DIR = DATA_DIR / "dataset"
SEED_FACTS = DATA_DIR / "seed_facts.jsonl"
SEED_GUARDRAILS = DATA_DIR / "seed_guardrails.jsonl"
SOURCES = DATA_DIR / "sources.json"

SPLITS = ("train", "validation", "test")
# Répartition stratifiée par source : (train, validation, test) en nombre de fiches.
CARD_SPLIT_STRATEGY = {
    "auscultation_cardiaque_normale": (30, 5, 5),
    "esc_evaluation_cardiovasculaire": (38, 6, 6),
}
GUARDRAIL_SPLIT = {"train": 42, "validation": 4, "test": 4}

FORMATS = ("qa_fr", "qa_en", "patient_fr", "patient_en", "reverse_fr", "reverse_en")

INSTRUCTIONS = {
    "qa_fr": (
        "Tu es un assistant pédagogique en cardiologie. Réponds à la question de "
        "façon concise et fidèle aux sources, sans poser de diagnostic."
    ),
    "qa_en": (
        "You are an educational cardiology assistant. Answer concisely and "
        "faithfully to the sources, without making a diagnosis."
    ),
    "patient_fr": (
        "Tu es un assistant pédagogique en cardiologie. Explique ce point à un "
        "patient en termes simples, sans poser de diagnostic ni prescrire."
    ),
    "patient_en": (
        "You are an educational cardiology assistant. Explain this point to a "
        "patient in simple terms, without diagnosing or prescribing."
    ),
    "reverse_fr": (
        "Tu es un assistant pédagogique en cardiologie. À partir de cette réponse, "
        "formule la question correspondante, sans poser de diagnostic."
    ),
    "reverse_en": (
        "You are an educational cardiology assistant. From this answer, write the "
        "corresponding question, without making a diagnosis."
    ),
}

REQUIRED_CARD_FIELDS = ("id", "source_id", "topic", "q_fr", "a_fr", "q_en", "a_en")
REQUIRED_GUARDRAIL_FIELDS = ("id", "lang", "topic", "instruction", "input", "output")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: JSON invalide ({error})") from error
    return records


def load_sources() -> dict:
    with SOURCES.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_cards(cards: list[dict], sources: dict) -> None:
    known = {source["id"] for source in sources["sources"]} | {"guardrails"}
    seen = set()
    for card in cards:
        missing = [field for field in REQUIRED_CARD_FIELDS if not card.get(field)]
        if missing:
            raise ValueError(f"Fiche {card.get('id', '?')}: champs manquants {missing}")
        if card["id"] in seen:
            raise ValueError(f"Fiche dupliquée: {card['id']}")
        seen.add(card["id"])
        if card["source_id"] not in known:
            raise ValueError(f"Fiche {card['id']}: source inconnue {card['source_id']}")
        if card["source_id"] == "guardrails":
            raise ValueError(f"Fiche {card['id']}: les guardrails doivent être dans seed_guardrails.jsonl")


def validate_guardrails(guardrails: list[dict]) -> None:
    seen = set()
    for item in guardrails:
        missing = [field for field in REQUIRED_GUARDRAIL_FIELDS if not item.get(field)]
        if missing:
            raise ValueError(f"Guardrail {item.get('id', '?')}: champs manquants {missing}")
        if item["id"] in seen:
            raise ValueError(f"Guardrail dupliqué: {item['id']}")
        seen.add(item["id"])
        if item["lang"] not in {"fr", "en"}:
            raise ValueError(f"Guardrail {item['id']}: langue invalide {item['lang']}")


def assign_card_splits(cards: list[dict], seed: int = 42) -> dict[str, str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_source[card["source_id"]].append(card)

    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for source_id, source_cards in by_source.items():
        if source_id not in CARD_SPLIT_STRATEGY:
            raise ValueError(f"Source sans stratégie de split: {source_id}")
        expected = sum(CARD_SPLIT_STRATEGY[source_id])
        if len(source_cards) != expected:
            raise ValueError(
                f"Source {source_id}: {len(source_cards)} fiches, {expected} attendues "
                f"par la stratégie de split."
            )
        shuffled = source_cards[:]
        rng.shuffle(shuffled)
        train_n, val_n, _ = CARD_SPLIT_STRATEGY[source_id]
        for index, card in enumerate(shuffled):
            if index < train_n:
                assignment[card["id"]] = "train"
            elif index < train_n + val_n:
                assignment[card["id"]] = "validation"
            else:
                assignment[card["id"]] = "test"
    return assignment


def assign_guardrail_splits(guardrails: list[dict], seed: int = 43) -> dict[str, str]:
    expected = sum(GUARDRAIL_SPLIT.values())
    if len(guardrails) != expected:
        raise ValueError(f"{len(guardrails)} guardrails, {expected} attendus.")
    rng = random.Random(seed)
    french = [item for item in guardrails if item["lang"] == "fr"]
    english = [item for item in guardrails if item["lang"] == "en"]
    rng.shuffle(french)
    rng.shuffle(english)

    # Alternance FR/EN pour que chaque split contienne les deux langues.
    interleaved: list[dict] = []
    for fr_item, en_item in zip(french, english):
        interleaved.extend([fr_item, en_item])
    interleaved.extend(french[len(english) :])
    interleaved.extend(english[len(french) :])

    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for item in interleaved[cursor : cursor + GUARDRAIL_SPLIT[split]]:
            assignment[item["id"]] = split
        cursor += GUARDRAIL_SPLIT[split]
    return assignment


def make_record(
    *,
    record_id: str,
    instruction: str,
    input_text: str,
    output: str,
    card: dict,
    lang: str,
    fmt: str,
    split: str,
    license_name: str,
) -> dict:
    return {
        "id": record_id,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "meta": {
            "card_id": card.get("card_id", card["id"]),
            "topic": card["topic"],
            "source_ids": card.get("source_ids", [card.get("source_id", "guardrails")]),
            "license": license_name,
            "lang": lang,
            "format": fmt,
            "split": split,
            "evidence": card.get("evidence", ""),
            "input_sha256": sha256(input_text),
            "output_sha256": sha256(output),
        },
    }


def build(sources: dict) -> dict[str, list[dict]]:
    cards = load_jsonl(SEED_FACTS)
    guardrails = load_jsonl(SEED_GUARDRAILS)
    validate_cards(cards, sources)
    validate_guardrails(guardrails)

    card_splits = assign_card_splits(cards)
    guardrail_splits = assign_guardrail_splits(guardrails)

    license_by_source = {source["id"]: source["license"] for source in sources["sources"]}
    license_by_source["guardrails"] = sources["guardrails"]["license"]

    records: dict[str, list[dict]] = {split: [] for split in SPLITS}

    for card in cards:
        split = card_splits[card["id"]]
        license_name = license_by_source[card["source_id"]]
        variants = {
            "qa_fr": (card["q_fr"], card["a_fr"]),
            "qa_en": (card["q_en"], card["a_en"]),
            "patient_fr": (card["q_fr"], card["a_fr"]),
            "patient_en": (card["q_en"], card["a_en"]),
            "reverse_fr": (card["a_fr"], card["q_fr"]),
            "reverse_en": (card["a_en"], card["q_en"]),
        }
        for fmt in FORMATS:
            input_text, output = variants[fmt]
            lang = "fr" if fmt.endswith("_fr") else "en"
            records[split].append(
                make_record(
                    record_id=f"{card['id']}::{fmt}",
                    instruction=INSTRUCTIONS[fmt],
                    input_text=input_text,
                    output=output,
                    card=card,
                    lang=lang,
                    fmt=fmt,
                    split=split,
                    license_name=license_name,
                )
            )

    for item in guardrails:
        split = guardrail_splits[item["id"]]
        records[split].append(
            make_record(
                record_id=f"{item['id']}::guardrail",
                instruction=item["instruction"],
                input_text=item["input"],
                output=item["output"],
                card={"id": item["id"], "topic": item["topic"], "source_id": "guardrails"},
                lang=item["lang"],
                fmt="guardrail",
                split=split,
                license_name=license_by_source["guardrails"],
            )
        )

    for split in SPLITS:
        records[split].sort(key=lambda record: record["id"])
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataset_stats(records: dict[str, list[dict]]) -> dict:
    stats = {
        "version": "1.0.0",
        "total": sum(len(items) for items in records.values()),
        "splits": {},
    }
    for split in SPLITS:
        items = records[split]
        stats["splits"][split] = {
            "total": len(items),
            "by_lang": dict(Counter(item["meta"]["lang"] for item in items)),
            "by_format": dict(Counter(item["meta"]["format"] for item in items)),
            "by_source": dict(Counter(item["meta"]["source_ids"][0] for item in items)),
            "unique_cards": len({item["meta"]["card_id"] for item in items}),
        }
    return stats


def write_dataset_card(stats: dict, sources: dict) -> None:
    lines = [
        "# Dataset Card — CardioAgent LLM (fr/en)",
        "",
        "> ⚠️ Corpus **pédagogique**. Il ne constitue ni un dispositif médical ni une aide au",
        "> diagnostic. Ne pas l'utiliser pour une décision clinique.",
        "",
        "## Résumé",
        "",
        f"- Version : `{stats['version']}`",
        f"- Total d'exemples : **{stats['total']}**",
        "- Format : JSONL, champs `instruction`, `input`, `output` (+ métadonnées `meta`).",
        "- Langues : français et anglais.",
        "- Tâches : question/réponse, explication au patient, question inverse, guardrails.",
        "",
        "## Répartition",
        "",
        "| Split | Total | FR | EN | Fiches uniques |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, values in stats["splits"].items():
        lines.append(
            f"| {split} | {values['total']} | {values['by_lang'].get('fr', 0)} | "
            f"{values['by_lang'].get('en', 0)} | {values['unique_cards']} |"
        )
    lines += [
        "",
        "## Sources",
        "",
        "| ID | Titre | Licence | Fichier |",
        "|---|---|---|---|",
    ]
    for source in sources["sources"]:
        lines.append(
            f"| `{source['id']}` | {source['title']} | {source['license']} | "
            f"`{source['file']}` |"
        )
    lines += [
        "",
        "## Méthode de construction",
        "",
        "1. Extraction du texte des deux PDF fournis (`pdftotext -layout`).",
        "2. Rédaction manuelle de **90 fiches de faits** ancrées, avec extrait justificatif.",
        "3. Génération déterministe de 6 tâches par fiche (FR/EN).",
        "4. Ajout de 50 guardrails de sécurité (FR/EN), sans donnée médicale nouvelle.",
        "5. Découpage stratifié par source et par fiche : toutes les variantes d'une fiche",
        "   restent dans le même split pour éviter les fuites.",
        "",
        "## Limites",
        "",
        "- Le corpus couvre deux documents seulement : il est volontairement restreint.",
        "- Les réponses sont des reformulations pédagogiques, pas des recommandations de prise",
        "  en charge individuelles.",
        "- Le cours d'auscultation a des droits non explicités : usage académique interne.",
        "- L'article est sous licence CC BY-NC 4.0 : usage non commercial avec attribution.",
        "- Aucune validation clinique n'a été réalisée.",
        "",
        "## Reproductibilité",
        "",
        "```bash",
        "python llm_finetuning/data/build_dataset.py",
        "pytest llm_finetuning/tests -q",
        "```",
    ]
    (DATASET_DIR / "dataset_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_existing() -> int:
    expected = {"train": 450, "validation": 70, "test": 70}
    errors = 0
    for split, count in expected.items():
        path = DATASET_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"[ERREUR] {path} absent")
            errors += 1
            continue
        records = load_jsonl(path)
        if len(records) != count:
            print(f"[ERREUR] {split}: {len(records)} exemples, {count} attendus")
            errors += 1
        for record in records:
            if not all(record.get(field) for field in ("instruction", "input", "output")):
                print(f"[ERREUR] {split}: exemple incomplet {record.get('id')}")
                errors += 1
    if errors == 0:
        print("[OK] Dataset présent et conforme aux volumes cibles.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Vérifie le dataset existant sans le régénérer.")
    args = parser.parse_args()

    if args.check:
        return 1 if check_existing() else 0

    sources = load_sources()
    records = build(sources)
    for split in SPLITS:
        write_jsonl(DATASET_DIR / f"{split}.jsonl", records[split])

    stats = dataset_stats(records)
    (DATASET_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "seed": {"cards": 42, "guardrails": 43},
        "splits": {
            split: [record["meta"]["card_id"] for record in records[split]]
            for split in SPLITS
        },
    }
    (DATASET_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_dataset_card(stats, sources)

    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))
    print(f"Total: {stats['total']} exemples -> {DATASET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
