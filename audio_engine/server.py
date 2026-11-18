"""audio-engine : prétraitement + inférence (ONNX historique + modèles TP3).

API REST consommée par l'interface web (via l'api-gateway) :

- ``GET  /health``        état du modèle ONNX et des trois modèles du TP3 ;
- ``GET  /models``        modèles disponibles (seuil, run MLflow, métriques) ;
- ``POST /predict``       modèle ONNX unique (``models/cardio_model.onnx``) ;
- ``POST /predict_all``   SVM + arbre de décision + CNN, avec consensus ;
- ``POST /predict_one``   un seul modèle (``?family=svm|tree|cnn``).

Les trois modèles du TP3 sont décrits dans ``registry.py`` : chacun est un artefact
local (``models/best_svm.joblib``, ``best_tree.joblib``, ``best_cnn.pt``) accompagné
de son JSON de métadonnées. MLflow sert uniquement à retrouver le run associé
(run_id, métriques) ; l'inférence ne dépend jamais du serveur MLflow.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import registry
from preprocess import AudioError, Preprocessor

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "cardio_model.onnx"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (ROOT / MODEL_PATH).resolve()
META_PATH = MODEL_PATH.with_name("cardio_model_meta.json")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
STATE: dict = {}


class PredictResponse(BaseModel):
    filename: str
    model: str
    duration_s: float
    n_segments: int
    prob_abnormal: float = Field(ge=0, le=1, description="moyenne des probabilités par segment")
    prob_abnormal_max: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    prediction: str
    segment_probs: List[float]
    features: Dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle ONNX puis les trois modèles du TP3 (best effort)."""
    try:
        import onnxruntime as ort

        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        STATE.update(meta=meta,
                     session=ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"]),
                     pre=Preprocessor(meta.get("preprocessing")))
        print(f"[audio-engine] ONNX chargé : {meta['model']} ({MODEL_PATH.name})")
    except Exception as error:                     # le service démarre quand même
        print(f"[audio-engine] ONNX non chargé : {type(error).__name__}: {error}")

    for key in registry.MODELS:
        try:
            registry.load_model(key)
            print(f"[audio-engine] modèle TP3 chargé : {key}")
        except Exception as error:
            print(f"[audio-engine] modèle TP3 indisponible ({key}) : {type(error).__name__}: {error}")
    yield


app = FastAPI(title="audio-engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(data: bytes) -> None:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Fichier trop volumineux (max {MAX_BYTES // (1024 * 1024)} Mo)")


def _check_wav(file: UploadFile) -> None:
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")


@app.get("/")
def root():
    return {"service": "audio-engine", "endpoints":
            ["/health", "/models", "/predict", "/predict_all", "/predict_one", "/docs"]}


@app.get("/health")
def health():
    tp3 = registry.status()
    return {
        "status": "ok",
        "onnx": {"loaded": bool(STATE), "model": STATE.get("meta", {}).get("model")},
        "models_tp3": {key: value["available"] for key, value in tp3.items()},
        "experiment": registry.EXPERIMENT,
    }


@app.get("/models")
def models():
    """Modèles du TP3 disponibles pour l'interface (seuil, run MLflow, métriques)."""
    return {
        "experiment": registry.EXPERIMENT,
        "mlflow_tracking_uri": registry.MLFLOW_URI,
        "models": registry.status(),
    }


@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...),
                      families: Optional[str] = Query(None, description="ex. svm,tree,cnn")):
    """Applique les trois modèles du TP3 au signal et renvoie le consensus."""
    _check_wav(file)
    data = await file.read()
    _read_upload(data)
    selected = [f.strip().lower() for f in families.split(",")] if families else None
    try:
        return registry.analyze(data, file.filename or "", selected)
    except AudioError as error:
        raise HTTPException(422, str(error))
    except ValueError as error:
        raise HTTPException(400, str(error))
    except Exception as error:
        raise HTTPException(500, f"Analyse multi-modèles impossible : {type(error).__name__}: {error}")


@app.post("/predict_one")
async def predict_one(file: UploadFile = File(...),
                      family: str = Query(..., description="svm | tree | cnn")):
    """Applique un seul modèle du TP3 (utile pour comparer les algorithmes)."""
    key = family.strip().lower()
    if key not in registry.MODELS:
        raise HTTPException(400, f"Famille inconnue : {family} (attendu : {', '.join(registry.MODELS)})")
    _check_wav(file)
    data = await file.read()
    _read_upload(data)
    try:
        report = registry.analyze(data, file.filename or "", [key])
    except AudioError as error:
        raise HTTPException(422, str(error))
    except Exception as error:
        raise HTTPException(500, f"Inférence impossible : {type(error).__name__}: {error}")
    if not report["analysis"]:
        raise HTTPException(500, "Aucun résultat")
    row = report["analysis"][0]
    row.update({"filename": report["filename"], "duration_s": report["duration_s"],
                "sample_rate_used": report["sample_rate_used"],
                "n_segments": report["n_segments"],
                "feature_summary": report["feature_summary"],
                "mlflow_tracking_uri_used": report["mlflow_tracking_uri_used"]})
    return row


def log_inference(resp: PredictResponse, latency_s: float):
    """Trace l'inférence ONNX dans MLflow (sans effet si MLflow est indisponible)."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri or os.getenv("MLFLOW_LOGGING", "1") != "1":
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(uri)
        exp = client.get_experiment_by_name("cardio-inference")
        exp_id = exp.experiment_id if exp else client.create_experiment("cardio-inference")
        run = client.create_run(exp_id, tags={"model": resp.model, "prediction": resp.prediction})
        for key, value in {"prob_abnormal": resp.prob_abnormal,
                           "prob_abnormal_max": resp.prob_abnormal_max,
                           "n_segments": resp.n_segments, "duration_s": resp.duration_s,
                           "latency_s": latency_s}.items():
            client.log_metric(run.info.run_id, key, float(value))
        client.set_terminated(run.info.run_id)
    except Exception as error:
        print(f"[audio-engine] log MLflow ignoré : {error}")


@app.post("/predict", response_model=PredictResponse)
async def predict(background: BackgroundTasks, file: UploadFile = File(...)):
    """Modèle ONNX unique (inchangé) — conservé pour compatibilité."""
    if not STATE:
        raise HTTPException(503, "Modèle ONNX non chargé")
    _check_wav(file)
    data = await file.read()
    _read_upload(data)
    t0 = time.time()
    pre, meta, session = STATE["pre"], STATE["meta"], STATE["session"]
    try:
        signal = pre.load_clean(data)
    except AudioError as error:
        raise HTTPException(422, str(error))
    segments = pre.segment(signal)
    kind = "features" if meta["input"]["type"] == "features" else "mel"
    outputs = session.run(None, {meta["input"]["name"]: pre.batch(segments, kind)})
    probs = np.asarray(outputs[1] if kind == "features" else outputs[0])[:, 1]
    p_mean, threshold = float(probs.mean()), float(meta["threshold"])
    features = np.stack([pre.features_1d(s) for s in segments]).mean(axis=0)
    names = pre.names
    summary = {"zcr": float(features[names.index("zcr_mean")]),
               "spectral_centroid_hz": float(features[names.index("centroid_mean")]),
               "rms": float(features[names.index("rms_mean")])}
    response = PredictResponse(
        filename=file.filename or "", model=meta["model"], duration_s=round(len(signal) / pre.sr, 2),
        n_segments=len(segments), prob_abnormal=p_mean, prob_abnormal_max=float(probs.max()),
        threshold=threshold, prediction="abnormal" if p_mean >= threshold else "normal",
        segment_probs=[round(float(p), 4) for p in probs], features=summary)
    background.add_task(log_inference, response, time.time() - t0)
    return response
