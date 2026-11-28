"""Tests de l'audio-engine : prétraitement, validation et API REST.

Aucun modèle réel n'est chargé : le prétraitement est testé sur des signaux
synthétiques et l'API utilise un faux session ONNX. Les notebooks, les poids et
le pipeline d'entraînement ne sont pas modifiés.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import registry
import server
from preprocess import DEFAULT_CFG, AudioError, Preprocessor, feature_names

SR = DEFAULT_CFG["sample_rate"]


def wav_bytes(seconds: float = 11.0, freq: float = 100.0, amplitude: float = 0.5) -> bytes:
    """WAV mono synthétique lisible par librosa (aucun fichier sur disque)."""
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, signal, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --- Prétraitement -----------------------------------------------------------


def test_feature_names_et_dimensions():
    names = feature_names(20)
    assert len(names) == 46
    assert names[0] == "mfcc0_mean"
    assert names[20] == "mfcc0_std"
    assert names[-1] == "rms_std"

    pre = Preprocessor()
    assert pre.names == names
    assert pre.sr == 2000
    assert pre.seg_len == 10_000
    assert pre.hop_len == 5_000


def test_load_clean_normalise_et_reéchantillonne():
    pre = Preprocessor()
    # Signal à 4000 Hz : librosa doit le ramener à 2000 Hz.
    t = np.arange(4000 * 4, dtype=np.float32) / 4000
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 80 * t), 4000, format="WAV", subtype="PCM_16")

    signal = pre.load_clean(buffer.getvalue())
    assert signal.dtype == np.float32
    assert abs(len(signal) / pre.sr - 4.0) < 0.05
    assert np.max(np.abs(signal)) <= 1.0 + 1e-6
    assert np.max(np.abs(signal)) > 0.5


def test_load_clean_refuse_audio_invalide():
    pre = Preprocessor()
    with pytest.raises(AudioError, match="illisible"):
        pre.load_clean(b"ceci n'est pas un wav")
    with pytest.raises(AudioError, match="trop court"):
        pre.load_clean(wav_bytes(seconds=0.5))
    with pytest.raises(AudioError, match="silence"):
        pre.load_clean(wav_bytes(amplitude=0.0))


def test_segmentation_avec_chevauchement():
    pre = Preprocessor()
    signal = pre.load_clean(wav_bytes(seconds=11.0))
    segments = pre.segment(signal)
    assert len(segments) == 3
    assert all(len(segment) == pre.seg_len for segment in segments)
    # 50 % de chevauchement : deux fenêtres consécutives partagent la moitié du signal.
    assert np.array_equal(segments[0][pre.hop_len :], segments[1][: pre.seg_len - pre.hop_len])


def test_segment_court_est_complete_et_long_est_plafonne():
    pre = Preprocessor()
    short = pre.load_clean(wav_bytes(seconds=3.0))
    assert len(pre.segment(short)) == 1
    assert len(pre.segment(short)[0]) == pre.seg_len

    capped = Preprocessor({"segment_sec": 1, "hop_sec": 0.5, "max_segments": 3})
    long = capped.load_clean(wav_bytes(seconds=8.0))
    segments = capped.segment(long)
    assert len(segments) == 3
    assert all(len(segment) == capped.seg_len for segment in segments)


def test_caracteristiques_et_lots():
    pre = Preprocessor()
    signal = pre.load_clean(wav_bytes(seconds=6.0))
    segment = pre.segment(signal)[0]

    features = pre.features_1d(segment)
    assert features.shape == (46,)
    assert np.isfinite(features).all()

    mel = pre.mel_db(segment)
    assert mel.shape[0] == pre.n_mels
    assert mel.shape[1] > 0
    assert np.isfinite(mel).all()

    batch_features = pre.batch([segment, segment], "features")
    assert batch_features.shape == (2, 46)
    assert batch_features.dtype == np.float32

    batch_mel = pre.batch([segment, segment], "mel")
    assert batch_mel.shape == (2, 1, pre.n_mels, mel.shape[1])
    assert batch_mel.dtype == np.float32


def test_config_invalide_est_ignoree():
    # Les clés inconnues ne doivent pas casser l'instance (payload de métadonnées).
    pre = Preprocessor({"inconnu": 42, "n_mfcc": 13})
    assert pre.n_mfcc == 13
    assert len(pre.names) == 32


# --- API REST ----------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """Client sans lifespan : aucun poids n'est chargé pendant les tests."""
    monkeypatch.setattr(registry, "mlflow_runs", lambda *args, **kwargs: {})
    server.STATE.clear()
    return TestClient(server.app)


def test_racine_et_health(client):
    root = client.get("/").json()
    assert root["service"] == "audio-engine"
    assert "/predict_all" in root["endpoints"]

    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["onnx"]["loaded"] is False
    assert set(body["models_tp3"]) == {"svm", "tree", "cnn", "tl", "hybrid"}


def test_models_expose_les_metadonnees(client):
    body = client.get("/models").json()
    assert body["experiment"] == registry.EXPERIMENT
    assert "svm" in body["models"]
    assert "available" in body["models"]["svm"]


def test_predict_sans_modele_renvoie_503(client):
    response = client.post("/predict", files={"file": ("test.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 503


class FakeSession:
    """Faux session ONNX : deux sorties dont la seconde contient les probabilités."""

    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [0.6, 0.8, 0.4]
        self.feeds = []

    def run(self, output_names, feed):
        self.feeds.append(feed)
        probs = np.asarray(self.probs, dtype=np.float32)
        logits = np.stack([1 - probs, probs], axis=1)
        return [np.zeros((len(probs), 2), dtype=np.float32), logits]


def _state_features():
    pre = Preprocessor()
    return {
        "pre": pre,
        "meta": {"model": "fake-onnx", "threshold": 0.5,
                 "input": {"type": "features", "name": "input"}},
        "session": FakeSession(),
    }


def test_predict_avec_faux_modele(client, monkeypatch):
    monkeypatch.setattr(server, "log_inference", lambda *args, **kwargs: None)
    server.STATE.update(_state_features())

    response = client.post("/predict", files={"file": ("test.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-onnx"
    assert body["prediction"] == "abnormal"  # moyenne 0,6 >= seuil 0,5
    assert body["prob_abnormal"] == pytest.approx(np.mean([0.6, 0.8, 0.4]), abs=1e-6)
    assert body["prob_abnormal_max"] == pytest.approx(0.8)
    assert body["n_segments"] == 3
    assert body["duration_s"] == pytest.approx(11.0, abs=0.1)
    assert len(body["segment_probs"]) == 3
    assert set(body["features"]) == {"zcr", "spectral_centroid_hz", "rms"}
    assert server.STATE["session"].feeds  # le modèle a bien été appelé


def test_predict_valide_extension_et_taille(client, monkeypatch):
    response = client.post("/predict", files={"file": ("test.mp3", wav_bytes(), "audio/mpeg")})
    assert response.status_code == 503  # le modèle n'est pas chargé avant la validation

    server.STATE.update(_state_features())
    response = client.post("/predict", files={"file": ("test.mp3", wav_bytes(), "audio/mpeg")})
    assert response.status_code == 400
    assert "wav" in response.json()["detail"].lower()

    monkeypatch.setattr(server, "MAX_BYTES", 10)
    response = client.post("/predict", files={"file": ("test.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 413

    monkeypatch.setattr(server, "MAX_BYTES", 25 * 1024 * 1024)
    response = client.post("/predict", files={"file": ("test.wav", b"invalide", "audio/wav")})
    assert response.status_code == 422


def test_predict_all_et_predict_one_valident_les_entrees(client, monkeypatch):
    assert client.post("/predict_all", files={"file": ("x.mp3", b"x", "audio/mpeg")}).status_code == 400
    assert client.post("/predict_one", files={"file": ("x.wav", wav_bytes(), "audio/wav")},
                       params={"family": "inconnu"}).status_code == 400

    def raise_audio_error(*args, **kwargs):
        raise AudioError("Signal vide (silence)")

    monkeypatch.setattr(registry, "analyze", raise_audio_error)
    response = client.post("/predict_all", files={"file": ("x.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 422
    assert "silence" in response.json()["detail"]


def test_predict_one_renvoie_la_ligne_demandee(client, monkeypatch):
    monkeypatch.setattr(registry, "analyze", lambda *args, **kwargs: {
        "filename": "x.wav",
        "duration_s": 11.0,
        "sample_rate_used": 2000,
        "n_segments": 3,
        "feature_summary": {"rms": 0.5},
        "mlflow_tracking_uri_used": "local",
        "analysis": [{"family": "SVM", "label": "SVM", "prediction": "normal",
                      "prob_abnormal": 0.2, "threshold": 0.5}],
    })
    body = client.post("/predict_one", params={"family": "svm"},
                       files={"file": ("x.wav", wav_bytes(), "audio/wav")}).json()
    assert body["family"] == "SVM"
    assert body["filename"] == "x.wav"
    assert body["n_segments"] == 3


def test_predict_one_sans_resultat_renvoie_500(client, monkeypatch):
    monkeypatch.setattr(registry, "analyze", lambda *args, **kwargs: {
        "filename": "x.wav", "duration_s": 0, "sample_rate_used": SR, "n_segments": 0,
        "feature_summary": {}, "mlflow_tracking_uri_used": "local", "analysis": [],
    })
    response = client.post("/predict_one", params={"family": "svm"},
                           files={"file": ("x.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 500
    assert "Aucun résultat" in response.json()["detail"]
