"""Tests du pipeline RAG (chunking, embeddings, indexation, récupération).

Ces tests n'utilisent ni GPU ni modèle Hugging Face : un encodeur factice et un
Qdrant en mémoire suffisent à valider la logique. L'indexation réelle est
documentée dans ``rag_engine/README.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag_engine.chunk import build_chunks, chunk_pages, normalize, _split_long_text
from rag_engine.embed import load_embeddings, save_embeddings
from rag_engine.index_qdrant import collection_vector_size, index_chunks, make_client, point_id
from rag_engine.evaluate import check_case, evaluate, main as evaluate_main
from rag_engine.retrieve import NO_CONTEXT, format_context, retrieve


class FakeEncoder:
    """Encodeur déterministe : 3 dimensions, vecteurs unitaires."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            if "B1" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "ESC" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _fake_chunks():
    return [
        {
            "id": "chunk_b1",
            "text": "Le B1 est intense, grave, sourd et prolongé.",
            "source": "course.pdf",
            "source_id": "auscultation_cardiaque_normale",
            "page": 5,
            "chunk_index": 0,
            "license": "usage académique interne",
            "disclaimer": "Usage pédagogique. Pas un avis médical.",
        },
        {
            "id": "chunk_esc",
            "text": "Recommandations ESC sur l'évaluation cardiovasculaire.",
            "source": "esc.pdf",
            "source_id": "esc_evaluation_cardiovasculaire",
            "page": 2,
            "chunk_index": 0,
            "license": "CC BY-NC 4.0",
            "disclaimer": "Usage pédagogique. Pas un avis médical.",
        },
    ]


def test_normalize_preserve_contenu():
    assert normalize("A\xa0B   C\r\n D") == "A B C\nD"


def test_split_avec_recouvrement():
    parts = _split_long_text(" ".join(["mot"] * 100), max_chars=60, overlap=10)
    assert len(parts) > 1
    assert all(len(part) <= 120 for part in parts)


def test_chunking_pdf_sources_uniques():
    chunks = build_chunks()
    assert len(chunks) >= 50
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert {chunk["source_id"] for chunk in chunks} == {
        "auscultation_cardiaque_normale",
        "esc_evaluation_cardiovasculaire",
    }
    for chunk in chunks:
        assert {"text", "source", "page", "chunk_index", "license", "disclaimer"} <= chunk.keys()
        assert len(chunk["text"]) >= 40


def test_chunk_pages_ignore_entetes_courts():
    assert chunk_pages([{"source": "x.pdf", "page": 1, "text": "court"}]) == []


def test_embeddings_npz_roundtrip(tmp_path):
    chunks = _fake_chunks()
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    path = tmp_path / "emb.npz"
    save_embeddings(chunks, vectors, "fake/model", path)
    ids, loaded, model = load_embeddings(path)
    assert ids == ["chunk_b1", "chunk_esc"]
    assert model == "fake/model"
    np.testing.assert_allclose(loaded, vectors)


def test_index_et_retrieve_en_memoire():
    client = make_client(in_memory=True)
    chunks = _fake_chunks()
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    count = index_chunks(client, chunks, vectors, "fake/model", collection="test_rag", recreate=True)
    assert count == 2
    assert collection_vector_size(client, "test_rag") == 3
    assert point_id("chunk_b1") != point_id("chunk_esc")

    results = retrieve(
        "Parle-moi du B1",
        client=client,
        model=FakeEncoder(),
        collection="test_rag",
        top_k=2,
        min_score=0.5,
    )
    assert results[0]["chunk_id"] == "chunk_b1"
    assert results[0]["source_id"] == "auscultation_cardiaque_normale"
    assert results[0]["page"] == 5
    context = format_context(results)
    assert "auscultation_cardiaque_normale" in context
    assert "sourd" in context


def test_context_vide_sans_fabrication():
    assert format_context([]) == NO_CONTEXT
    assert "Aucun passage" in format_context([])


def test_score_threshold_filtre_les_resultats():
    client = make_client(in_memory=True)
    chunks = _fake_chunks()
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    index_chunks(client, chunks, vectors, "fake/model", collection="test_threshold", recreate=True)
    results = retrieve(
        "ESC",
        client=client,
        model=FakeEncoder(),
        collection="test_threshold",
        top_k=5,
        min_score=0.99,
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_esc"


def test_index_refuse_taille_de_vecteur_differente():
    client = make_client(in_memory=True)
    chunks = _fake_chunks()
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index_chunks(client, chunks, vectors, "fake/model", collection="test_size", recreate=True)
    with pytest.raises(ValueError, match="taille de vecteur"):
        index_chunks(
            client,
            chunks,
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "fake/model",
            collection="test_size",
        )


# --- Évaluation reproductible ------------------------------------------------


def _case(**kwargs):
    return {"id": "q1", "query": "B1 ?", "expected_source_id": "src", **kwargs}


def _result(**kwargs):
    return {
        "text": "Le B1 est intense et sourd.",
        "source_id": "src",
        "page": 5,
        "score": 0.8,
        "disclaimer": "Usage pédagogique",
        **kwargs,
    }


def test_check_case_valide_les_criteres():
    assert check_case(_case(keywords_any=["sourd"], expect_disclaimer=True), [_result()]) == []


def test_check_case_detecte_les_echecs():
    assert check_case(_case(expect_empty=True), [_result()])
    assert check_case(_case(), [])
    assert any("source attendue" in f for f in check_case(_case(), [_result(source_id="autre")]))
    assert any("mot-clé" in f for f in check_case(_case(keywords_any=["inexistant"]), [_result()]))
    assert any("avertissement" in f for f in check_case(
        _case(expect_disclaimer=True), [_result(disclaimer="")]
    ))


def test_evaluate_agrege_les_resultats(monkeypatch, tmp_path):
    import rag_engine.evaluate as module

    queries = tmp_path / "eval_queries.json"
    queries.write_text(json.dumps([_case(keywords_any=["sourd"])]) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "make_client", lambda url: object())
    monkeypatch.setattr(module, "load_model", lambda name: object())
    monkeypatch.setattr(module, "retrieve", lambda *args, **kwargs: [_result()])

    report = evaluate(queries_path=queries, url="http://test", collection="test")
    assert report["passed"] == 1
    assert report["total"] == 1
    assert report["details"][0]["status"] == "PASS"
    assert report["details"][0]["top_source_id"] == "src"
    assert report["details"][0]["top_page"] == 5
    assert report["details"][0]["n_passages"] == 1


def test_main_ecrit_le_rapport_et_le_code_de_sortie(monkeypatch, tmp_path):
    import sys

    import rag_engine.evaluate as module

    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(module, "evaluate", lambda *args, **kwargs: {
        "passed": 0, "total": 0, "details": [],
    })
    monkeypatch.setattr(sys, "argv", ["evaluate", "--queries", str(queries),
                                      "--output", str(output)])

    assert evaluate_main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] == 0
    assert report["total"] == 0
