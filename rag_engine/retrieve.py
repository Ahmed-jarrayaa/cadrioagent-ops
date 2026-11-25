"""Recherche de similarité dans la base vectorielle RAG.

Fournit les passages sources à l'orchestrateur : le LLM ne fait que reformuler,
il n'est jamais la source unique d'un fait médical. Si aucun passage n'est
trouvé, le contexte est vide et le message l'indique explicitement.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from rag_engine.embed import DEFAULT_MODEL, embed_query, load_model
from rag_engine.index_qdrant import DEFAULT_COLLECTION, DEFAULT_URL, make_client

NO_CONTEXT = "Aucun passage pertinent trouvé dans les documents sources."


def retrieve(
    query: str,
    top_k: int = 5,
    client=None,
    model=None,
    collection: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Retourne les passages les plus proches, avec score et provenance."""
    client = client if client is not None else make_client(DEFAULT_URL)
    model = model if model is not None else load_model(model_name)
    vector = embed_query(model, query)
    response = client.query_points(
        collection_name=collection,
        query=np.asarray(vector, dtype=np.float32).tolist(),
        limit=top_k,
        with_payload=True,
        score_threshold=min_score,
    )
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        results.append(
            {
                "score": round(float(point.score), 4),
                "text": payload.get("text", ""),
                "source": payload.get("source"),
                "source_id": payload.get("source_id"),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "license": payload.get("license"),
                "disclaimer": payload.get("disclaimer"),
            }
        )
    return results


def format_context(results: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Construit le contexte textuel injecté dans le prompt du LLM."""
    if not results:
        return NO_CONTEXT
    sections: list[str] = []
    total = 0
    for index, item in enumerate(results, start=1):
        header = (
            f"[{index}] Source : {item['source_id']} — {item['source']}, "
            f"page {item['page']} (similarité {item['score']:.3f})"
        )
        section = f"{header}\n{item['text']}"
        if total + len(section) > max_chars and sections:
            break
        sections.append(section)
        total += len(section)
    return "\n\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question en français ou en anglais")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()

    client = make_client(args.url)
    results = retrieve(
        args.query,
        top_k=args.top_k,
        client=client,
        model=load_model(args.model),
        collection=args.collection,
        model_name=args.model,
        min_score=args.min_score,
    )
    print(format_context(results))
    print(f"\n--- {len(results)} passage(s) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
