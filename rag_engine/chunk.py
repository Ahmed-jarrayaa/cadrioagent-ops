"""Découpage des PDF sources du projet en passages indexables.

Aucune donnée médicale n'est inventée : le texte provient uniquement des deux PDF
déclarés dans ``llm_finetuning/data/sources.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = REPO_ROOT / "llm_finetuning" / "data" / "sources.json"
DEFAULT_PDF_DIR = REPO_ROOT / "rag_engine"


def normalize(text: str) -> str:
    """Normalise les espaces sans altérer le contenu."""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extrait une entrée par page, avec le numéro de page (1-based)."""
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if text:
            pages.append({"source": pdf_path.name, "page": index, "text": text})
    return pages


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Découpe un texte en fenêtres délimitées par des mots."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        if size + len(word) + (1 if current else 0) > max_chars and current:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current = tail
            size = len(" ".join(current))
        current.append(word)
        size += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: Iterable[dict[str, Any]], max_chars: int = 900, overlap: int = 120
) -> list[dict[str, Any]]:
    """Produit des passages avec identifiant stable et métadonnées de provenance."""
    chunks: list[dict[str, Any]] = []
    for page in pages:
        for index, text in enumerate(_split_long_text(page["text"], max_chars, overlap)):
            if len(text.strip()) < 40:  # ignore les en-têtes/pieds de page isolés
                continue
            key = f"{page['source']}|{page['page']}|{index}|{text}"
            chunks.append(
                {
                    "id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
                    "text": text.strip(),
                    "source": page["source"],
                    "page": int(page["page"]),
                    "chunk_index": index,
                }
            )
    return chunks


def load_source_metadata() -> dict[str, dict[str, Any]]:
    """Associe chaque PDF à son identifiant et sa licence déclarés dans le projet."""
    data = json.loads(SOURCES.read_text(encoding="utf-8"))
    return {Path(item["file"]).name: item for item in data["sources"]}


def build_chunks(pdf_dir: Path = DEFAULT_PDF_DIR) -> list[dict[str, Any]]:
    """Extrait et découpe les deux PDF du projet."""
    metadata = load_source_metadata()
    chunks: list[dict[str, Any]] = []
    for filename, info in metadata.items():
        pdf_path = Path(pdf_dir) / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF source absent : {pdf_path}")
        for chunk in chunk_pages(extract_pages(pdf_path)):
            chunk["source_id"] = info["id"]
            chunk["license"] = info["license"]
            chunk["disclaimer"] = "Usage pédagogique. Pas un avis médical."
            chunks.append(chunk)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--output", default=None, help="Fichier JSONL de sortie (optionnel)")
    args = parser.parse_args()
    chunks = build_chunks(Path(args.input_dir))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
            encoding="utf-8",
        )
        print(f"[chunk] {len(chunks)} passages -> {output}")
    else:
        print(json.dumps({"n_chunks": len(chunks), "sources": sorted({c['source_id'] for c in chunks})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
