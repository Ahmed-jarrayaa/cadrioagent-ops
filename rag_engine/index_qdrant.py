"""Indexation des passages et embeddings dans Qdrant.

Le conteneur ``vector-db`` expose Qdrant sur ``http://127.0.0.1:6333``.
L'indexation est idempotente : les identifiants des passages sont stables (SHA-256
du contenu), donc relancer la commande remplace les points existants.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rag_engine.embed import DEFAULT_MODEL, embed_texts, load_embeddings, load_model, save_embeddings

DEFAULT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cardio_docs")
NAMESPACE = uuid.UUID("6f1a4c1e-8f3a-4d7b-9f24-1b6c9a5d0e42")


def make_client(url: str = DEFAULT_URL, in_memory: bool = False):
    """Retourne un client Qdrant (local HTTP ou mémoire pour les tests)."""
    from qdrant_client import QdrantClient

    return QdrantClient(":memory:") if in_memory else QdrantClient(url=url, timeout=60)


def point_id(chunk_id: str) -> str:
    """Identifiant UUID stable accepté par Qdrant."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def collection_vector_size(client, name: str) -> int | None:
    """Taille des vecteurs d'une collection existante, ou ``None``."""
    try:
        info = client.get_collection(name)
    except Exception:
        return None
    vectors = info.config.params.vectors
    return int(vectors.size) if hasattr(vectors, "size") else None


def ensure_collection(client, name: str, size: int, recreate: bool = False) -> None:
    """Crée la collection avec une distance cosinus, ou la remplace."""
    from qdrant_client import models

    exists = collection_vector_size(client, name) is not None
    if recreate and exists:
        client.delete_collection(name)
        exists = False
    if exists:
        current = collection_vector_size(client, name)
        if current != size:
            raise ValueError(
                f"Collection '{name}' : taille de vecteur {current} != {size}. "
                "Utiliser --recreate ou le même modèle d'embedding."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )


def payload_for(chunk: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Métadonnées de provenance attachées à chaque point."""
    return {
        "text": chunk["text"],
        "chunk_id": chunk["id"],
        "source": chunk["source"],
        "source_id": chunk["source_id"],
        "page": int(chunk["page"]),
        "chunk_index": int(chunk["chunk_index"]),
        "license": chunk["license"],
        "disclaimer": chunk["disclaimer"],
        "embedding_model": model_name,
    }


def index_chunks(
    client,
    chunks: Sequence[dict[str, Any]],
    vectors: np.ndarray,
    model_name: str,
    collection: str = DEFAULT_COLLECTION,
    recreate: bool = False,
    batch_size: int = 64,
) -> int:
    """Upserte tous les passages ; retourne le nombre de points indexés."""
    from qdrant_client import models

    if len(chunks) != len(vectors):
        raise ValueError("Nombre de passages et de vecteurs différent")
    if not chunks:
        raise ValueError("Aucun passage à indexer")

    ensure_collection(client, collection, int(vectors.shape[1]), recreate=recreate)
    points = [
        models.PointStruct(
            id=point_id(chunk["id"]),
            vector=vector.tolist(),
            payload=payload_for(chunk, model_name),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[start : start + batch_size], wait=True)
    return len(points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", default=None, help="JSONL de passages (défaut : extraction directe)")
    parser.add_argument("--embeddings", default=None, help="Fichier .npz produit par embed.py (évite de recalculer)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recreate", action="store_true", help="Supprime et recrée la collection")
    parser.add_argument("--in-memory", action="store_true", help="Qdrant en mémoire (tests/démo)")
    args = parser.parse_args()

    from rag_engine.chunk import build_chunks

    if args.embeddings:
        ids, vectors, model_name = load_embeddings(Path(args.embeddings))
        if args.chunks:
            chunks = [
                json.loads(line)
                for line in Path(args.chunks).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            chunks = build_chunks()
        by_id = {chunk["id"]: chunk for chunk in chunks}
        missing = [chunk_id for chunk_id in ids if chunk_id not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} identifiant(s) d'embedding sans passage correspondant")
        chunks = [by_id[chunk_id] for chunk_id in ids]
    else:
        chunks = build_chunks() if not args.chunks else [
            json.loads(line)
            for line in Path(args.chunks).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        model = load_model(args.model, args.device)
        vectors = embed_texts(model, [chunk["text"] for chunk in chunks])
        model_name = args.model
        from rag_engine.embed import DATA_DIR, model_slug

        save_embeddings(chunks, vectors, model_name, DATA_DIR / f"embeddings_{model_slug(model_name)}.npz")

    client = make_client(args.url, args.in_memory)
    count = index_chunks(client, chunks, vectors, model_name, args.collection, args.recreate)
    print(
        f"[index] collection={args.collection} points={count} dim={vectors.shape[1]} "
        f"modele={model_name} url={'memory' if args.in_memory else args.url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
