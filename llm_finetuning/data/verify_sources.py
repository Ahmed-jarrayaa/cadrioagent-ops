"""Vérifie la provenance des données : PDF fournis, empreintes et texte extrait.

Contrôles :
1. chaque PDF déclaré dans ``sources.json`` existe et son SHA-256 correspond ;
2. le texte extrait (``data/raw``) existe et n'est pas vide ;
3. si ``pdftotext`` est disponible, le texte ré-extrait est comparé au texte stocké.

Usage :
    python llm_finetuning/data/verify_sources.py
    python llm_finetuning/data/verify_sources.py --refresh   # ré-écrit data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]
SOURCES = DATA_DIR / "sources.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def token_overlap(reference: str, candidate: str) -> float:
    """Proportion de tokens du texte stocké présents dans le texte ré-extrait."""
    ref_tokens = normalize(reference).split()
    if not ref_tokens:
        return 0.0
    candidate_tokens = set(normalize(candidate).split())
    return sum(1 for token in ref_tokens if token in candidate_tokens) / len(ref_tokens)


def extract_pdf(pdf_path: Path) -> str | None:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out.txt"
        subprocess.run(
            [pdftotext, "-layout", str(pdf_path), str(output)],
            check=True,
            capture_output=True,
        )
        return output.read_text(encoding="utf-8", errors="ignore")


def verify(refresh: bool = False) -> int:
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    errors = 0

    for source in sources["sources"]:
        source_id = source["id"]
        pdf_path = REPO_ROOT / source["file"]
        text_path = REPO_ROOT / source["extracted_text"]

        if not pdf_path.exists():
            print(f"[ERREUR] {source_id}: PDF absent -> {pdf_path}")
            errors += 1
            continue

        actual = sha256_file(pdf_path)
        expected = source.get("sha256", "").lower()
        if not expected:
            print(f"[ERREUR] {source_id}: empreinte absente de sources.json")
            errors += 1
        elif actual != expected:
            print(
                f"[ERREUR] {source_id}: SHA-256 différent\n"
                f"         attendu  {expected}\n"
                f"         obtenu   {actual}"
            )
            errors += 1
        else:
            print(f"[OK] {source_id}: PDF et empreinte concordent")

        fresh = extract_pdf(pdf_path)
        if fresh is None:
            print(f"[INFO] {source_id}: pdftotext indisponible, comparaison ignorée")
            if not text_path.exists():
                print(f"[ERREUR] {source_id}: texte extrait absent -> {text_path}")
                errors += 1
            continue

        if refresh:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(fresh, encoding="utf-8", newline="\n")
            print(f"[OK] {source_id}: texte ré-extrait -> {text_path}")
            continue

        if not text_path.exists():
            print(f"[ERREUR] {source_id}: texte extrait absent -> {text_path}")
            errors += 1
            continue

        stored = text_path.read_text(encoding="utf-8", errors="ignore")
        if not normalize(stored):
            print(f"[ERREUR] {source_id}: texte extrait vide")
            errors += 1
            continue

        overlap = token_overlap(stored, fresh)
        if overlap < 0.98:
            print(f"[ERREUR] {source_id}: texte extrait incohérent (recouvrement {overlap:.1%})")
            errors += 1
        else:
            print(f"[OK] {source_id}: texte extrait cohérent (recouvrement {overlap:.1%})")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ré-extrait les PDF vers data/raw (au lieu de seulement comparer).",
    )
    args = parser.parse_args()
    errors = verify(refresh=args.refresh)
    if errors:
        print(f"[ÉCHEC] {errors} problème(s) de provenance.")
        return 1
    print("[OK] Provenance des sources vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
