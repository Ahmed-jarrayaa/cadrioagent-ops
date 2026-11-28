"""Tests de l'api-gateway : proxys, validation des uploads et gestion des pannes.

Le dossier ``api-gateway`` contient un tiret : il n'est pas importable comme un
paquet. Le module est donc chargé par chemin, puis ``httpx.AsyncClient`` est
remplacé par un transport factice. Aucun conteneur n'est requis.
"""

from __future__ import annotations

import importlib.util
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
    def __init__(self, body=None, status_code: int = 200, text: str | None = None):
        self._body = body
        self.status_code = status_code
        self.headers = {"content-type": "application/json" if body is not None else "text/plain"}
        self.text = text if text is not None else ("" if body is None else str(body))

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


def install_transport(monkeypatch, handler):
    """Remplace ``httpx.AsyncClient`` et retourne la liste des appels."""
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
            return handler("POST", url, kwargs)

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
