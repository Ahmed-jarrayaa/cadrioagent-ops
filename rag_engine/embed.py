"""Embeddings des passages RAG avec Sentence-Transformers.

Modèle par défaut : ``BAAI/bge-m3`` (multilingue FR/EN, 1024 dimensions).
Les vecteurs sont normalisés (norme L2 = 1) : Qdrant utilisera la similarité cosinus.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"
DATA_DIR = Path(__file__).resolve().parent / "data"


def model_slug(model_name: str) -> str:
    """Nom de fichier stable dérivé du nom du modèle Hugging Face."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Charge le modèle d'embedding (déjà en cache local si possible)."""
    from sentence_transformers import SentenceTransformer

    offline = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    kwargs: dict[str, Any] = {"local_files_only": offline}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def embed_texts(model, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Encode une liste de textes en vecteurs float32 normalisés."""
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(model, query: str) -> np.ndarray:
    """Encode une requête unique ; conserved pour symétrie avec ``embed_texts``."""
    return embed_texts(model, [query])[0]


def save_embeddings(
    chunks: Iterable[dict[str, Any]], vectors: np.ndarray, model_name: str, path: Path
) -> None:
    """Sauvegarde les vecteurs et leurs identifiants dans un fichier ``.npz``."""
    chunks = list(chunks)
    ids = np.array([chunk["id"] for chunk in chunks], dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=ids,
        vectors=np.asarray(vectors, dtype=np.float32),
        model=np.array(model_name),
    )


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Lit un fichier produit par ``save_embeddings``."""
    with np.load(path, allow_pickle=True) as data:
        return (
            [str(value) for value in data["ids"]],
            np.asarray(data["vectors"], dtype=np.float32),
            str(data["model"]),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=None, help="JSONL de passages (défaut : extraction directe des PDF)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="ex. cpu ou cuda (défaut : auto)")
    parser.add_argument("--output", default=None, help="Fichier .npz (défaut : rag_engine/data/embeddings_<modele>.npz)")
    args = parser.parse_args()

    from rag_engine.chunk import build_chunks

    if args.chunks:
        chunks = [
            json.loads(line)
            for line in Path(args.chunks).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        chunks = build_chunks()

    output = Path(args.output) if args.output else DATA_DIR / f"embeddings_{model_slug(args.model)}.npz"
    model = load_model(args.model, args.device)
    vectors = embed_texts(model, [chunk["text"] for chunk in chunks])
    save_embeddings(chunks, vectors, args.model, output)
    print(f"[embed] modèle={args.model} passages={len(chunks)} dim={vectors.shape[1]} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
