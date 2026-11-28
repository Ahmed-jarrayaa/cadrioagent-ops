"""Tests de l'api-gateway : proxys, validation des uploads et gestion des pannes.

Le dossier ``api-gateway`` contient un tiret : il n'est pas importable comme un
paquet. Le module est donc chargé par chemin, puis ``httpx.AsyncClient`` est
remplacé par un transport factice. Aucun conteneur n'est requis.
"""

from __future__ import annotations

import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("api_gateway_main", ROOT / "api-gateway" / "main.py")
gateway = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gateway)


class FakeResponse:
    def __init__(self, body=None, status_code: int = 200, text: str | None = None,
                 lines: list[str] | None = None):
        self._body = body
        self.status_code = status_code
        self.headers = {"content-type": "application/json" if body is not None else "text/plain"}
        self.text = text if text is not None else ("" if body is None else str(body))
        self._lines = lines or []

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        if self._body is None:
            raise ValueError("pas de JSON")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://upstream.test")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request,
                response=httpx.Response(self.status_code, request=request),
            )


PROMPT_OK = {
    "relevant": True, "language": "fr", "intent": "report",
    "reason": "ok", "message": "",
}


def install_transport(monkeypatch, handler, prompt_check=None):
    """Remplace ``httpx.AsyncClient`` et retourne la liste des appels.

    La vérification de pertinence est simulée comme réussie par défaut ; les
    tests peuvent la remplacer avec ``prompt_check``.
    """
    prompt_body = PROMPT_OK if prompt_check is None else prompt_check
    calls: list[tuple[str, str, dict]] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            return handler("GET", url, kwargs)

        async def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            if url.startswith(f"{gateway.ORCH}/check_prompt"):
                return FakeResponse(prompt_body)
            return handler("POST", url, kwargs)

        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            calls.append(("STREAM", url, kwargs))
            yield handler(method, url, kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", FakeAsyncClient)
    return calls


def test_racine_liste_les_endpoints():
    with TestClient(gateway.app) as client:
        body = client.get("/").json()
    assert body["service"] == "CardioAgent-Ops API Gateway"
    assert "/analyze" in body["endpoints"]
    assert "/analyze_pcg" in body["endpoints"]


def test_health_agrege_les_trois_services(monkeypatch):
    def handler(method, url, kwargs):
        return FakeResponse({"status": "ok", "url": url})

    calls = install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert set(body["services"]) == {"api-gateway", "audio-engine", "orchestrator", "mlflow"}
    assert all(body["services"][name]["status"] == "ok"
               for name in ("audio-engine", "orchestrator", "mlflow"))
    assert len(calls) == 3


def test_health_signale_un_service_hors_ligne(monkeypatch):
    def handler(method, url, kwargs):
        if url.startswith(gateway.AUDIO):
            raise RuntimeError("connexion refusée")
        return FakeResponse({"status": "ok"})

    install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["services"]["audio-engine"]["status"] == "down"
    assert "connexion refusée" in body["services"]["audio-engine"]["error"]
    assert body["services"]["orchestrator"]["status"] == "ok"


def test_analyze_transmet_le_payload(monkeypatch):
    expected = {"status": "refused", "intent": "diagnosis", "answer": "non"}
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse(expected))

    with TestClient(gateway.app) as client:
        response = client.post("/analyze", json={"question": "Diagnostique ce souffle."})

    assert response.status_code == 200
    assert response.json() == expected
    assert calls[0][0] == "POST"
    assert calls[0][1] == f"{gateway.ORCH}/analyze"
    assert calls[0][2]["json"] == {"question": "Diagnostique ce souffle."}


def test_analyze_propage_le_code_erreur(monkeypatch):
    install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse(
        {"detail": "requête invalide"}, status_code=422))
    with TestClient(gateway.app) as client:
        response = client.post("/analyze", json={"question": "B1 ?"})
    assert response.status_code == 422
    assert "requête invalide" in response.json()["detail"]


def test_analyze_orchestrateur_indisponible(monkeypatch):
    def handler(method, url, kwargs):
        raise RuntimeError("orchestrateur arrêté")

    install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        response = client.post("/analyze", json={"question": "B1 ?"})
    assert response.status_code == 502
    assert "indisponible" in response.json()["detail"]


def test_analyze_pcg_refuse_les_fichiers_non_wav(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse({}))
    with TestClient(gateway.app) as client:
        response = client.post("/analyze_pcg", files={"file": ("souffle.mp3", b"x", "audio/mpeg")})
    assert response.status_code == 400
    assert calls == []


def test_analyze_pcg_enchaine_audio_puis_orchestrateur(monkeypatch):
    prediction = {"prediction": "abnormal", "prob_abnormal": 0.81, "threshold": 0.5, "model": "onnx"}
    audited = {"status": "ok", "intent": "knowledge", "answer": "score de dépistage"}
    calls = install_transport(monkeypatch, lambda method, url, kwargs: (
        FakeResponse(prediction) if url.startswith(gateway.AUDIO) else FakeResponse(audited)
    ))

    with TestClient(gateway.app) as client:
        response = client.post("/analyze_pcg", files={"file": ("souffle.wav", b"RIFF", "audio/wav")})

    assert response.status_code == 200
    assert response.json() == audited
    assert [call[1] for call in calls] == [f"{gateway.AUDIO}/predict", f"{gateway.ORCH}/analyze"]
    assert calls[0][2]["files"]["file"][0] == "souffle.wav"
    assert calls[1][2]["json"] == prediction


def test_analyze_pcg_erreur_audio(monkeypatch):
    install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse(
        {"detail": "modèle non chargé"}, status_code=503))
    with TestClient(gateway.app, raise_server_exceptions=False) as client:
        response = client.post("/analyze_pcg", files={"file": ("x.wav", b"RIFF", "audio/wav")})
    assert response.status_code == 500


def test_predict_proxy_et_validation(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse({"prediction": "normal"}))

    with TestClient(gateway.app) as client:
        assert client.post("/predict", files={"file": ("x.txt", b"x", "text/plain")}).status_code == 400
        assert calls == []

        response = client.post("/predict", files={"file": ("x.wav", b"RIFF", "audio/wav")})

    assert response.status_code == 200
    assert response.json()["prediction"] == "normal"
    assert calls[0][1] == f"{gateway.AUDIO}/predict"


def test_analyze_all_transmet_les_familles(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse({"analysis": []}))
    with TestClient(gateway.app) as client:
        response = client.post("/analyze_all?families=svm,tree",
                               files={"file": ("x.wav", b"RIFF", "audio/wav")})
    assert response.status_code == 200
    assert calls[0][1] == f"{gateway.AUDIO}/predict_all"
    assert calls[0][2]["params"] == {"families": "svm,tree"}


def test_predict_one_famille_par_defaut(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse({"family": "SVM"}))
    with TestClient(gateway.app) as client:
        response = client.post("/predict_one", files={"file": ("x.wav", b"RIFF", "audio/wav")})
    assert response.status_code == 200
    assert calls[0][1] == f"{gateway.AUDIO}/predict_one"
    assert calls[0][2]["params"] == {"family": "svm"}


def test_models_indisponible(monkeypatch):
    def handler(method, url, kwargs):
        raise RuntimeError("audio-engine arrêté")

    install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        response = client.get("/models")
    assert response.status_code == 502
    assert "audio-engine indisponible" in response.json()["detail"]


def test_mlflow_experiments(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse({"experiments": []}))
    with TestClient(gateway.app) as client:
        response = client.get("/mlflow/experiments")
    assert response.status_code == 200
    assert response.json() == {"experiments": []}
    assert calls[0][1] == f"{gateway.MLFLOW}/api/2.0/mlflow/experiments/search"
    assert calls[0][2]["params"] == {"max_results": 50}


# --- Workflow 5 modèles + rapport -------------------------------------------


WORKFLOW = {
    "filename": "signal.wav",
    "analysis": [
        {"key": key, "label": key.upper(), "status": "ok", "prediction": "abnormal",
         "prob_abnormal_mean": 0.8, "prob_abnormal_max": 0.9, "threshold": 0.5}
        for key in ("svm", "tree")
    ] + [
        {"key": key, "label": key.upper(), "status": "ok", "prediction": "normal",
         "prob_abnormal_mean": 0.2, "prob_abnormal_max": 0.3, "threshold": 0.5}
        for key in ("cnn", "tl")
    ] + [
        {"key": "hybrid", "label": "HYBRID", "status": "error", "error": "artefact absent"}
    ],
    "consensus": {"prediction": "abnormal", "votes": {"normal": 2, "abnormal": 2},
                  "models_ok": 4, "models_total": 5, "mean_prob_abnormal": 0.5,
                  "mean_threshold": 0.5},
    "feature_summary": {"zcr_mean": 0.12, "spectral_centroid_hz": 88.4, "rms_mean": 0.03},
    "duration_s": 5.0,
    "n_segments": 1,
    "sample_rate_used": 2000,
    "preprocessing": {"bandpass_hz": [25, 400]},
}
REPORT = {"status": "ok", "intent": "report", "answer": "Rapport", "sources": []}


def test_analyze_all_report_enchaine_5_modeles_puis_agent(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: (
        FakeResponse(WORKFLOW) if url.startswith(gateway.AUDIO) else FakeResponse(REPORT)
    ))

    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
            data={"instruction": "Analyse ce signal et génère un rapport détaillé."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Rapport"
    assert body["instruction"] == "Analyse ce signal et génère un rapport détaillé."
    assert body["workflow"] == WORKFLOW
    assert [call[1] for call in calls] == [
        f"{gateway.ORCH}/check_prompt", f"{gateway.AUDIO}/predict_all",
        f"{gateway.ORCH}/analyze",
    ]
    assert calls[1][2]["params"]["families"] == "svm,tree,cnn,tl,hybrid"
    payload = calls[2][2]["json"]
    assert payload["mode"] == "report"
    assert payload["question"] == "Analyse ce signal et génère un rapport détaillé."
    assert payload["workflow"]["analysis"][0]["key"] == "svm"
    assert payload["prediction"] == "abnormal"
    assert payload["prob_abnormal"] == 0.5
    trace = body["workflow_trace"]
    assert [item["service"] for item in trace] == list(gateway.SERVICE_ORDER)
    assert trace[1]["status"] in {"ok", "degrade"}
    assert trace[4]["detail"] == "0 passage(s) utilisé(s)"
    assert body["gateway_latency_ms"] >= 0


def test_workflow_services_expose_les_six_services(monkeypatch):
    def handler(method, url, kwargs):
        if url.startswith(gateway.ORCH):
            return FakeResponse({"services": {
                "qdrant": {"status": "ok", "points": 66, "collection": "cardio_docs"},
                "llm": {"status": "ok", "model": "cardio-llm"},
            }})
        return FakeResponse({"status": "ok"})

    calls = install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        body = client.get("/workflow/services").json()

    assert body["order"] == list(gateway.SERVICE_ORDER)
    assert set(body["services"]) == set(gateway.SERVICE_ORDER)
    assert body["services"]["api-gateway"]["status"] == "ok"
    assert body["services"]["vector-db"]["points"] == 66
    assert body["services"]["llm-server"]["model"] == "cardio-llm"
    assert len(calls) == 3


def test_prompt_hors_contexte_bloque_avant_toute_prediction(monkeypatch):
    rejected = {
        "relevant": False, "language": "fr", "intent": "out_of_scope",
        "reason": "out_of_scope",
        "message": "Cette demande n\u2019est pas liée au domaine du système. Reformulez votre prompt.",
    }
    calls = install_transport(
        monkeypatch,
        lambda method, url, kwargs: FakeResponse(WORKFLOW),
        prompt_check=rejected,
    )
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
            data={"instruction": "Recette du gateau"},
        )
    assert response.status_code == 400
    assert "Reformulez votre prompt" in response.json()["detail"]
    assert [call[1] for call in calls] == [f"{gateway.ORCH}/check_prompt"]


def test_analyze_all_report_bloque_si_un_modele_manque(monkeypatch):
    incomplete = dict(WORKFLOW, analysis=WORKFLOW["analysis"][:4])
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse(incomplete))
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report", files={"file": ("signal.wav", b"RIFF", "audio/wav")}
        )
    assert response.status_code == 502
    assert "hybrid" in response.json()["detail"]
    assert [call[1] for call in calls] == [
        f"{gateway.ORCH}/check_prompt", f"{gateway.AUDIO}/predict_all"
    ]  # l'agent n'est jamais appelé


def test_analyze_all_report_valide_fichier_et_instruction(monkeypatch):
    calls = install_transport(monkeypatch, lambda method, url, kwargs: FakeResponse(WORKFLOW))
    with TestClient(gateway.app) as client:
        assert client.post(
            "/analyze_all_report", files={"file": ("signal.mp3", b"x", "audio/mpeg")}
        ).status_code == 400
        assert client.post(
            "/analyze_all_report",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
            data={"instruction": "   "},
        ).status_code == 400
    assert calls == []


# --- Flux SSE temps réel ------------------------------------------------------


def parse_sse(text):
    """Décode un corps SSE en liste de couples (event, data)."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        events.append((event, data))
    return events


def orchestrator_stream_lines(report):
    lines = []
    for event, payload in [
        ("step", {"service": "vector-db", "status": "running", "detail": "recherche"}),
        ("step", {"service": "vector-db", "status": "ok", "detail": "3 passages"}),
        ("step", {"service": "llm-server", "status": "running", "detail": "confirmation"}),
        ("step", {"service": "llm-server", "status": "ok", "detail": "validée"}),
        ("step", {"service": "agent-orchestrator", "status": "ok", "detail": "audit ok"}),
        ("result", report),
    ]:
        lines += [f"event: {event}", "data: " + json.dumps(payload, ensure_ascii=False), ""]
    return lines


def test_stream_relaye_les_etapes_reelles_et_trace_dans_mlflow(monkeypatch):
    report = dict(REPORT, sources=[{"source": "doc", "page": 1, "score": 0.8}])
    lines = orchestrator_stream_lines(report)

    def handler(method, url, kwargs):
        if url.startswith(gateway.AUDIO):
            return FakeResponse(WORKFLOW)
        if url.startswith(gateway.MLFLOW):
            if "experiments/search" in url:
                return FakeResponse({"experiments": [
                    {"name": "cardio-workflow", "experiment_id": "7"}
                ]})
            if "runs/create" in url:
                return FakeResponse({"run": {"info": {"run_id": "run-9"}}})
            return FakeResponse({})
        if "/analyze/stream" in url:
            return FakeResponse(lines=lines)
        return FakeResponse(report)

    calls = install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report/stream",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
            data={"instruction": "Analyse ce signal et conclus."},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = parse_sse(response.text)
    steps = [(item["service"], item["status"])
             for event, item in events if event == "step"]
    assert steps == [
        ("api-gateway", "running"), ("api-gateway", "ok"),
        ("audio-engine", "running"), ("audio-engine", "degrade"),
        ("mlflow-server", "running"), ("mlflow-server", "ok"),
        ("agent-orchestrator", "running"),
        ("vector-db", "running"), ("vector-db", "ok"),
        ("llm-server", "running"), ("llm-server", "ok"),
        ("agent-orchestrator", "ok"),
    ]
    result = [item for event, item in events if event == "result"][0]
    assert result["workflow"] == WORKFLOW
    assert result["instruction"] == "Analyse ce signal et conclus."
    assert len(result["workflow_trace"]) == 6
    assert [item for event, item in events if event == "status"][0]["status"] == "terminated"

    urls = [call[1] for call in calls]
    assert f"{gateway.ORCH}/analyze/stream" in urls
    assert f"{gateway.MLFLOW}/api/2.0/mlflow/runs/create" in urls
    assert f"{gateway.MLFLOW}/api/2.0/mlflow/runs/update" in urls
    assert any("log-metric" in url for url in urls)
    assert any("step.vector-db" in str(call[2]) for call in calls)


def test_stream_arrete_au_service_indispensable_en_erreur(monkeypatch):
    calls = install_transport(
        monkeypatch,
        lambda method, url, kwargs: FakeResponse({"detail": "panne"}, status_code=503),
    )
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report/stream",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
        )

    events = parse_sse(response.text)
    errors = [item for event, item in events if event == "error"]
    assert errors and errors[0]["service"] == "audio-engine"
    assert errors[0]["indispensable"] is True
    assert not any(event == "result" for event, _ in events)
    assert [call[1] for call in calls] == [
        f"{gateway.ORCH}/check_prompt", f"{gateway.AUDIO}/predict_all"
    ]


def test_stream_bloque_prompt_hors_domaine_avant_prediction(monkeypatch):
    rejected = {
        "relevant": False, "language": "fr", "intent": "out_of_scope",
        "reason": "out_of_scope",
        "message": "Cette demande n\u2019est pas liée au domaine du système. Reformulez votre prompt.",
    }
    calls = install_transport(
        monkeypatch,
        lambda method, url, kwargs: FakeResponse(WORKFLOW),
        prompt_check=rejected,
    )
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report/stream",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
            data={"instruction": "Donne-moi la recette du gateau."},
        )

    events = parse_sse(response.text)
    steps = [(item["service"], item["status"])
             for event, item in events if event == "step"]
    assert steps == [("api-gateway", "running"), ("api-gateway", "erreur")]
    errors = [item for event, item in events if event == "error"]
    assert errors[0]["service"] == "api-gateway"
    assert "Reformulez votre prompt" in errors[0]["detail"]
    assert not any(event == "result" for event, _ in events)
    assert [call[1] for call in calls] == [f"{gateway.ORCH}/check_prompt"]


def test_stream_repli_sur_analyze_si_orchestrateur_sans_sse(monkeypatch):
    report = dict(REPORT, sources=[{"source": "doc", "page": 2, "score": 0.7}],
                  audit={"passed": True, "warnings": []})

    def handler(method, url, kwargs):
        if url.startswith(gateway.AUDIO):
            return FakeResponse(WORKFLOW)
        if url.startswith(gateway.MLFLOW):
            if "experiments/search" in url:
                return FakeResponse({"experiments": []})
            if "experiments/create" in url:
                return FakeResponse({"experiment_id": "4"})
            if "runs/create" in url:
                return FakeResponse({"run": {"info": {"run_id": "run-2"}}})
            return FakeResponse({})
        if method == "STREAM":
            return FakeResponse(status_code=404)
        return FakeResponse(report)

    install_transport(monkeypatch, handler)
    with TestClient(gateway.app) as client:
        response = client.post(
            "/analyze_all_report/stream",
            files={"file": ("signal.wav", b"RIFF", "audio/wav")},
        )

    events = parse_sse(response.text)
    steps = [(item["service"], item["status"])
             for event, item in events if event == "step"]
    assert ("vector-db", "ok") in steps
    assert ("llm-server", "ok") in steps
    assert any(event == "result" for event, _ in events)
