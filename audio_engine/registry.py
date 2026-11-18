"""Registre des trois modèles du TP3 (SVM, arbre de décision, CNN) pour l'inférence.

Chaque modèle est un couple « artefact local + métadonnées JSON » produit par son notebook :

    models/best_svm.joblib   + best_svm_meta.json    Pipeline(StandardScaler, SVC) → decision_function
    models/best_tree.joblib  + best_tree_meta.json   DecisionTreeClassifier          → predict_proba
    models/best_cnn.pt       + best_cnn_meta.json    CardioCNNLite (state_dict)      → softmax

Le module est utilisé par :
- ``server.py`` (service audio-engine : ``/predict_all``, ``/predict_one``, ``/models``) ;
- ``analyze_signal.py`` (CLI) ;
- les tests / le notebook d'évaluation.

Les métadonnées MLflow (run_id, run_name, métriques) sont lues dans l'expérience
``tp3_audio_engine`` quand le serveur est joignable ; sinon le registre fonctionne
uniquement avec les fichiers locaux (important pour le TP hors ligne).

Usage :
    python registry.py chemin/vers.wav --json
    python registry.py chemin/vers.wav --families svm,cnn
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import AudioError, Preprocessor  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("TP3_EXPERIMENT", "tp3_audio_engine")
CLASSES = ["normal", "abnormal"]

#: Description des trois familles servies par l'interface.
MODELS: Dict[str, dict] = {
    "svm": {
        "family": "SVM",
        "label": "SVM (rbf)",
        "experiment": EXPERIMENT,
        "run_name": "svm_rbf_segments",
        "model_path": MODEL_DIR / "best_svm.joblib",
        "meta_path": MODEL_DIR / "best_svm_meta.json",
        "decision": {"kind": "decision_function", "threshold": 0.5, "probability": "sigmoid",
                     "score_threshold": 0.0},
    },
    "tree": {
        "family": "DecisionTree",
        "label": "Arbre de décision",
        "experiment": EXPERIMENT,
        "run_name": "decision_tree_gridsearch",
        "model_path": MODEL_DIR / "best_tree.joblib",
        "meta_path": MODEL_DIR / "best_tree_meta.json",
        "decision": {"kind": "predict_proba", "threshold": 0.5},
    },
    "cnn": {
        "family": "CNN",
        "label": "CNN (Mel)",
        "experiment": EXPERIMENT,
        "run_name": "cnn_mel_segments",
        "model_path": MODEL_DIR / "best_cnn.pt",
        "meta_path": MODEL_DIR / "best_cnn_meta.json",
        "decision": {"kind": "softmax", "threshold": 0.5},
    },
}

DEFAULT_MEL = {"sr": 2000, "n_mels": 128, "n_fft": 512, "hop_length": 128,
               "target_time_frames": 256, "fmin": 20, "fmax": 1000}

_cache: Dict[str, object] = {}
_mlflow_cache: Dict[str, dict] = {"at": 0.0, "runs": {}}


# --------------------------------------------------------------- utilitaires
def _read_meta(spec: dict) -> dict:
    try:
        return json.loads(spec["meta_path"].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sigmoid(scores):
    return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))


def mel_spectrogram(segment: np.ndarray, cfg: Optional[dict] = None) -> np.ndarray:
    """Mel 128×256 dB normalisé par segment — identique à l'entraînement du CNN."""
    import librosa

    c = {**DEFAULT_MEL, **(cfg or {})}
    m = librosa.feature.melspectrogram(
        y=segment.astype(np.float32), sr=int(c["sr"]), n_fft=int(c["n_fft"]),
        hop_length=int(c["hop_length"]), n_mels=int(c["n_mels"]),
        fmin=float(c.get("fmin", 20)), fmax=float(c.get("fmax", int(c["sr"]) // 2)),
    )
    m = librosa.power_to_db(m, ref=np.max)
    target = int(c["target_time_frames"])
    m = np.pad(m, ((0, 0), (0, max(0, target - m.shape[1]))))[:, :target]
    m = m.astype(np.float32)
    return (m - m.mean()) / (m.std() + 1e-6)


# ------------------------------------------------------ découverte MLflow
def mlflow_runs(force: bool = False, ttl_s: float = 60.0) -> dict:
    """Dernier run de chaque famille dans l'expérience partagée (cache 60 s)."""
    if not force and time.time() - _mlflow_cache["at"] < ttl_s:
        return _mlflow_cache["runs"]

    runs: Dict[str, dict] = {}
    # Le serveur MLflow n'écoute parfois que sur une des deux piles (IPv4/IPv6) :
    # on essaie l'URI configurée puis son équivalent localhost/127.0.0.1.
    candidates = [MLFLOW_URI]
    for a, b in (("http://127.0.0.1", "http://localhost"), ("http://localhost", "http://127.0.0.1")):
        if MLFLOW_URI.startswith(a):
            candidates.append(b + MLFLOW_URI[len(a):])
    errors = []
    for uri in candidates:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(uri)
            client = MlflowClient()
            exp = client.get_experiment_by_name(EXPERIMENT)
        except Exception as error:                 # serveur MLflow arrêté : mode local
            errors.append(f"{uri} -> {type(error).__name__}: {error}")
            continue
        try:
            if exp is not None:
                found = client.search_runs([exp.experiment_id],
                                           order_by=["attributes.start_time DESC"],
                                           max_results=100)
                for spec in MODELS.values():
                    for run in found:
                        if run.data.tags.get("mlflow.runName") == spec["run_name"]:
                            runs[spec["run_name"]] = {
                                "run_id": run.info.run_id,
                                "run_name": spec["run_name"],
                                "experiment": EXPERIMENT,
                                "experiment_id": exp.experiment_id,
                                "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                                "params": dict(run.data.params),
                                "artifacts": [a.path for a in
                                              client.list_artifacts(run.info.run_id, "models")],
                            }
                            break
            else:
                errors.append(f"{uri} -> expérience '{EXPERIMENT}' absente")
        except Exception as error:                 # métadonnées indisponibles : mode local
            errors.append(f"{uri} -> {type(error).__name__}: {error}")
            continue
        break
    else:
        runs = {"_error": {"error": " | ".join(errors)}}

    _mlflow_cache.update(at=time.time(), runs=runs)
    return runs


# ------------------------------------------------------------- chargement
def load_model(key: str):
    """Charge (et met en cache) le modèle d'une famille : 'svm', 'tree' ou 'cnn'.

    Le cache est invalidé si l'artefact a été réécrit (notebook relancé pendant que le
    service tourne) : ``/predict_all`` sert donc toujours les derniers poids entraînés.
    """
    if key not in MODELS:
        raise KeyError(f"Famille inconnue : {key} (attendu : {', '.join(MODELS)})")

    spec = MODELS[key]
    path = Path(spec["model_path"])
    cached = _cache.get(key)
    if cached and cached.get("mtime") == (path.stat().st_mtime if path.exists() else None):
        return cached

    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} absent : lancer le notebook {spec['label']} pour l'entraîner "
            f"(la section MLflow produit l'artefact).")

    meta = _read_meta(spec)
    if key == "cnn":
        import torch

        from src.model_cnn_lite import CardioCNNLite

        model = CardioCNNLite(n_classes=len(CLASSES))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        artifact = {"model": model, "meta": meta, "device": "cpu"}
    else:
        import joblib

        artifact = {"model": joblib.load(path), "meta": meta}
    artifact["mtime"] = path.stat().st_mtime
    _cache[key] = artifact
    return artifact


def status() -> dict:
    """Disponibilité de chaque famille (sans charger les poids)."""
    runs = mlflow_runs()
    out = {}
    for key, spec in MODELS.items():
        model_path = Path(spec["model_path"])
        meta = _read_meta(spec)
        run = runs.get(spec["run_name"], {})
        out[key] = {
            "family": spec["family"],
            "label": spec["label"],
            "available": model_path.exists(),
            "model_file": model_path.name if model_path.exists() else None,
            "size_mb": round(model_path.stat().st_size / 1e6, 3) if model_path.exists() else None,
            "threshold": float(meta.get("threshold", spec["decision"]["threshold"])),
            "mlflow_run_id": run.get("run_id"),
            "mlflow_run_name": run.get("run_name"),
            "experiment": spec["experiment"],
            "metrics": {k: run.get("metrics", {}).get(k) for k in
                        ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1", "roc_auc")},
        }
    return out


# ------------------------------------------------------------ prédiction
def predict_segments(key: str, segments: Iterable[np.ndarray], pre: Optional[Preprocessor] = None):
    """P(abnormal) pour chaque segment d'un enregistrement.

    Returns:
        (probabilities, threshold, info) — ``probabilities`` est toujours une probabilité
        (le score SVM est passé dans une sigmoïde, transformée monotone du ``decision_function``).
    """
    spec = MODELS[key]
    artifact = load_model(key)
    meta = artifact["meta"] or {}
    segments = list(segments)
    if not segments:
        raise AudioError("Aucun segment exploitable dans ce signal")

    threshold = meta.get("threshold")
    if threshold is None:
        threshold = spec["decision"]["threshold"]
    threshold = float(threshold)

    if key == "svm":
        pre = pre or Preprocessor()
        features = np.stack([pre.features_1d(s) for s in segments]).astype(np.float32)
        pipeline = artifact["model"]
        if hasattr(pipeline, "decision_function"):
            probs = _sigmoid(pipeline.decision_function(features))
        else:                                        # SVC(probability=True) éventuel
            probs = pipeline.predict_proba(features)[:, 1]
    elif key == "tree":
        pre = pre or Preprocessor()
        features = np.stack([pre.features_1d(s) for s in segments]).astype(np.float32)
        probs = artifact["model"].predict_proba(features)[:, 1]
    elif key == "cnn":
        import torch

        mel_cfg = meta.get("mel", DEFAULT_MEL)
        mels = np.stack([mel_spectrogram(s, mel_cfg) for s in segments])[:, None, :, :]
        model = artifact["model"]
        with torch.no_grad():
            logits = model(torch.from_numpy(mels.astype(np.float32)))
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    else:                                            # pragma: no cover
        raise KeyError(key)

    probs = np.asarray(probs, dtype=float)
    info = {
        "threshold": threshold,
        "decision": spec["decision"],
        "n_features": int(len(meta.get("features", {}).get("names", []))) or None,
    }
    return probs, threshold, info


def analyze(data: bytes, filename: str = "", families: Optional[List[str]] = None,
            pre: Optional[Preprocessor] = None) -> dict:
    """Analyse multi-modèles d'un WAV : rapport JSON consommé par l'interface web."""
    pre = pre or Preprocessor()
    signal = pre.load_clean(data)
    segments = pre.segment(signal)

    selected = [k for k in MODELS if families is None or k in families]
    if not selected:
        raise ValueError(f"Aucune famille valide : {families}")

    runs = mlflow_runs()
    features = np.stack([pre.features_1d(s) for s in segments])
    feat_mean = features.mean(axis=0)
    names = pre.names
    summary = {}
    for label, name in (("zcr_mean", "zcr_mean"), ("spectral_centroid_hz", "centroid_mean"),
                        ("rms_mean", "rms_mean")):
        if name in names:
            summary[label] = float(feat_mean[names.index(name)])

    results = []
    for key in selected:
        spec = MODELS[key]
        run = runs.get(spec["run_name"], {})
        row = {"family": spec["family"], "key": key, "label": spec["label"],
               "experiment": spec["experiment"], "mlflow_run_id": run.get("run_id"),
               "mlflow_run_name": run.get("run_name"), "status": "ok", "error": None}
        try:
            probs, threshold, info = predict_segments(key, segments, pre)
            p_mean = float(probs.mean())
            meta = load_model(key)["meta"] or {}
            reference = meta.get("mean_metrics") or (meta.get("metrics", {}) or {}).get("test") or {}
            row.update({
                "threshold": threshold,
                "prob_abnormal_mean": p_mean,
                "prob_abnormal_max": float(probs.max()),
                "prob_abnormal_min": float(probs.min()),
                "prediction": "abnormal" if p_mean >= threshold else "normal",
                "n_segments": int(len(probs)),
                "segment_probs": [round(float(p), 6) for p in probs],
                "training_metrics_reference": {
                    k: reference.get(k) for k in
                    ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1", "roc_auc")
                    if isinstance(reference.get(k), (int, float))
                },
                "mlflow_metrics": {k: run.get("metrics", {}).get(k) for k in
                                   ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")},
            })
        except Exception as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    consensus = None
    if ok:
        votes = {"normal": 0, "abnormal": 0}
        for row in ok:
            votes[row["prediction"]] += 1
        mean_prob = float(np.mean([r["prob_abnormal_mean"] for r in ok]))
        mean_threshold = float(np.mean([r["threshold"] for r in ok]))
        if votes["normal"] == votes["abnormal"]:      # égalité : on tranche sur la probabilité moyenne
            prediction = "abnormal" if mean_prob >= mean_threshold else "normal"
        else:
            prediction = max(votes, key=votes.get)
        consensus = {
            "prediction": prediction,
            "votes": votes,
            "models_ok": len(ok),
            "models_total": len(results),
            "mean_prob_abnormal": mean_prob,
            "mean_threshold": mean_threshold,
        }

    return {
        "filename": filename,
        "duration_s": round(len(signal) / pre.sr, 3),
        "sample_rate_used": pre.sr,
        "n_segments": len(segments),
        "feature_summary": summary,
        "mlflow_tracking_uri_used": MLFLOW_URI,
        "experiment": EXPERIMENT,
        "models_discovered_from_mlflow": [
            {"family": k, "experiment": EXPERIMENT, "run_id": v.get("run_id"),
             "run_name": v.get("run_name"), "metrics": v.get("metrics", {})}
            for k, v in runs.items() if k in {s["run_name"] for s in MODELS.values()}
        ],
        "analysis": results,
        "consensus": consensus,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyse multi-modèles d'un WAV PCG")
    parser.add_argument("audio", help="chemin du fichier .wav")
    parser.add_argument("--json", action="store_true", help="sortie JSON complète")
    parser.add_argument("--families", default=None,
                        help="sous-ensemble : svm,tree,cnn (défaut : les trois)")
    args = parser.parse_args()

    path = Path(args.audio)
    families = [f.strip() for f in args.families.split(",")] if args.families else None
    report = analyze(path.read_bytes(), path.name, families)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=" * 74)
    print("ANALYSE PCG — SVM / arbre de décision / CNN (audio_engine)")
    print("=" * 74)
    print(f"Fichier  : {report['filename']}")
    print(f"Durée    : {report['duration_s']} s | {report['n_segments']} segment(s) | "
          f"sr={report['sample_rate_used']} Hz")
    print(f"MLflow   : {report['mlflow_tracking_uri_used']} / {report['experiment']}")
    print("-" * 74)
    for row in report["analysis"]:
        if row["status"] != "ok":
            print(f"[{row['label']}] ERREUR : {row['error']}")
            continue
        print(f"[{row['label']}] run={row['mlflow_run_name']} ({row['mlflow_run_id']})")
        print(f"   prédiction={row['prediction']}  P(abnormal)={row['prob_abnormal_mean']:.3f}  "
              f"seuil={row['threshold']:.3f}  max={row['prob_abnormal_max']:.3f}")
    print("-" * 74)
    c = report["consensus"]
    if c:
        print(f"CONSENSUS : {c['prediction']}  votes={c['votes']}  "
              f"modèles OK={c['models_ok']}/{c['models_total']}  "
              f"P(abnormal) moyenne={c['mean_prob_abnormal']:.3f}")


if __name__ == "__main__":
    main()
