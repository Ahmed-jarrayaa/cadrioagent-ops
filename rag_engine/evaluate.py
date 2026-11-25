"""Évaluation reproductible du rappel RAG.

Pour chaque question de ``eval_queries.json``, on vérifie la source dominante,
la présence de mots-clés dans les passages, le seuil de confiance et
l'avertissement pédagogique. Le rapport JSON est versionné comme preuve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_engine.embed import DEFAULT_MODEL, load_model
from rag_engine.index_qdrant import DEFAULT_COLLECTION, DEFAULT_URL, make_client
from rag_engine.retrieve import retrieve

HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = HERE / "eval_queries.json"
DEFAULT_REPORT = HERE / "reports" / "rag_evaluation.json"


def check_case(case: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    """Retourne la liste des vérifications en échec."""
    failures: list[str] = []
    if case.get("expect_empty"):
        if results:
            failures.append(f"attendu aucun passage, obtenu {len(results)} (score max {results[0]['score']})")
        return failures
    if not results:
        failures.append("aucun passage au-dessus du seuil")
        return failures
    expected = case.get("expected_source_id")
    if expected and results[0]["source_id"] != expected:
        failures.append(f"source attendue {expected}, obtenue {results[0]['source_id']}")
    keywords = case.get("keywords_any") or []
    if keywords:
        text = " ".join(item["text"] for item in results).lower()
        if not any(keyword.lower() in text for keyword in keywords):
            failures.append(f"aucun mot-clé parmi {keywords}")
    if case.get("expect_disclaimer") and any(not item["disclaimer"] for item in results):
        failures.append("avertissement pédagogique absent d'un passage")
    return failures


def evaluate(
    queries_path: Path = DEFAULT_QUERIES,
    url: str = DEFAULT_URL,
    collection: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
) -> dict[str, Any]:
    cases = json.loads(queries_path.read_text(encoding="utf-8"))
    client = make_client(url)
    model = load_model(model_name)
    details: list[dict[str, Any]] = []
    for case in cases:
        results = retrieve(
            case["query"],
            top_k=top_k,
            client=client,
            model=model,
            collection=collection,
            model_name=model_name,
            min_score=float(case.get("min_score", 0.35)),
        )
        failures = check_case(case, results)
        details.append(
            {
                "id": case["id"],
                "query": case["query"],
                "description": case.get("description", ""),
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
                "top_score": results[0]["score"] if results else None,
                "top_source_id": results[0]["source_id"] if results else None,
                "top_page": results[0]["page"] if results else None,
                "n_passages": len(results),
                "snippet": results[0]["text"][:220] if results else "",
            }
        )
    passed = sum(item["status"] == "PASS" for item in details)
    return {
        "collection": collection,
        "model": model_name,
        "url": url,
        "top_k": top_k,
        "passed": passed,
        "total": len(details),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    report = evaluate(
        Path(args.queries), args.url, args.collection, args.model, args.top_k
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in report["details"]:
        print(f"[{item['status']}] {item['id']} — score={item['top_score']} source={item['top_source_id']}")
        for failure in item["failures"]:
            print(f"    - {failure}")
    print(f"Résultat : {report['passed']}/{report['total']} — rapport {output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
