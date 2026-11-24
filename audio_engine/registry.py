"""Registre des cinq modèles servis par l'audio-engine.

Chaque modèle est un couple « artefact local + métadonnées JSON » produit par son notebook :

    models/best_svm.joblib   + best_svm_meta.json    Pipeline(StandardScaler, SVC) → decision_function
    models/best_tree.joblib  + best_tree_meta.json   DecisionTreeClassifier          → predict_proba
    models/best_cnn.keras    + best_cnn_meta.json    CNN TensorFlow/Keras            → softmax
    models/best_tl_<tag>.keras + best_tl_<tag>_meta.json
        Transfer learning (ResNet18 / ResNet50 / VGG16) : Mel 64×128 → 128×128 RGB
        → ``preprocess_input`` du backbone → softmax. Le meilleur backbone (AUC OOF)
        est sélectionné automatiquement ; ``TL_BACKBONE`` permet de le forcer.
    models/best_transformer_cnn_hybrid.keras + best_transformer_cnn_hybrid_meta.json
        CNN + Transformer Encoder sur Mel 64×128 (channels_last) → softmax.

Le module est utilisé par :
- ``server.py`` (service audio-engine : ``/predict_all``, ``/predict_one``, ``/models``) ;
- ``analyze_signal.py`` (CLI) ;
- les tests / le notebook d'évaluation.

Les métadonnées MLflow (run_id, run_name, métriques) sont lues dans l'expérience
``tp3_audio_engine`` (serveur MLflow, puis store SQLite local ``mlruns.db``). Si un
artefact local a été supprimé, :func:`restore_from_mlflow` le télécharge depuis le
run MLflow correspondant : pas besoin de relancer le notebook pour un modèle déjà
entraîné. L'inférence reste utilisable hors ligne une fois les fichiers restaurés.

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
#: Store MLflow local utilisé par les notebooks quand le serveur n'est pas lancé.
_LOCAL_DB = ROOT / "mlruns.db"

#: Description des cinq familles servies par l'interface.
MODELS: Dict[str, dict] = {
    "svm": {
        "family": "SVM",
        "label": "SVM (rbf)",
        "kind": "sklearn",
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
        "kind": "sklearn",
        "experiment": EXPERIMENT,
        "run_name": "decision_tree_gridsearch",
        "model_path": MODEL_DIR / "best_tree.joblib",
        "meta_path": MODEL_DIR / "best_tree_meta.json",
        "decision": {"kind": "predict_proba", "threshold": 0.5},
    },
    "cnn": {
        "family": "CNN",
        "label": "CNN (Mel)",
        "kind": "keras",
        "experiment": EXPERIMENT,
        "run_name": "cnn_mel_segments",
        "model_path": MODEL_DIR / "best_cnn.keras",
        "meta_path": MODEL_DIR / "best_cnn_meta.json",
        "decision": {"kind": "softmax", "threshold": 0.5},
    },
    "tl": {
        "family": "TransferLearning",
        "label": "Transfer Learning",
        "kind": "keras",
        "experiment": EXPERIMENT,
        # run_name résolu dynamiquement : tl_<backbone>_3fold
        "run_name": None,
        "model_path": None,          # résolu parmi best_tl_<tag>.keras
        "meta_path": None,
        "decision": {"kind": "softmax", "threshold": 0.5},
    },
    "hybrid": {
        "family": "TransformerCNN",
        "label": "Transformer-CNN (hybride)",
        "kind": "keras",
        "experiment": EXPERIMENT,
        "run_name": "transformer_cnn_hybrid_3fold",
        "model_path": MODEL_DIR / "best_transformer_cnn_hybrid.keras",
        "meta_path": MODEL_DIR / "best_transformer_cnn_hybrid_meta.json",
        "decision": {"kind": "softmax", "threshold": 0.5},
    },
}

DEFAULT_MEL = {"sr": 2000, "n_mels": 128, "n_fft": 512, "hop_length": 128,
               "target_time_frames": 256, "fmin": 20, "fmax": 1000}

_cache: Dict[str, object] = {}
_mlflow_cache: Dict[str, dict] = {"at": 0.0, "runs": {}}
_hybrid_layer_cls = None


# --------------------------------------------------------------- utilitaires
def _preload_tensorflow() -> None:
    """Charge TensorFlow avant MLflow (ordre d'import obligatoire sur Windows).

    Si ``mlflow`` est importé en premier, ``import tensorflow`` échoue ensuite avec
    ``ImportError: DLL load failed while importing _pywrap_tensorflow_internal``.
    Le reste du module fonctionne sans TensorFlow : on ignore donc son absence.
    """
    if "tensorflow" in sys.modules:
        return
    try:
        import tensorflow  # noqa: F401
    except Exception:
        pass


def _read_meta_file(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_meta(spec: dict) -> dict:
    return _read_meta_file(spec["meta_path"])


def _meta_auc(meta: dict) -> Optional[float]:
    """AUC OOF d'un méta JSON, quel que soit le notebook qui l'a produit."""
    candidates = [
        (meta.get("mean_metrics") or {}).get("auc"),
        ((meta.get("metrics") or {}).get("oof_segment") or {}).get("auc"),
        ((meta.get("metrics") or {}).get("test") or {}).get("roc_auc"),
        (meta.get("metrics") or {}).get("roc_auc"),
    ]
    for value in candidates:
        try:
            if value is not None and np.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def tl_candidates() -> List[dict]:
    """Modèles ``best_tl_<tag>.keras`` présents dans ``MODEL_DIR``, avec leur AUC OOF."""
    out = []
    for meta_path in sorted(MODEL_DIR.glob("best_tl_*_meta.json")):
        tag = meta_path.name[len("best_tl_"):-len("_meta.json")]
        model_path = MODEL_DIR / f"best_tl_{tag}.keras"
        if not model_path.exists():
            continue
        meta = _read_meta_file(meta_path)
        out.append({"tag": tag,
                    "backbone": meta.get("backbone") or tag.upper(),
                    "model_path": model_path,
                    "meta_path": meta_path,
                    "meta": meta,
                    "auc": _meta_auc(meta)})
    return out


def resolve_tl() -> Optional[dict]:
    """Meilleur modèle de transfer learning (AUC OOF) ; ``TL_BACKBONE`` force un backbone."""
    candidates = tl_candidates()
    if not candidates:
        return None
    forced = os.getenv("TL_BACKBONE", "").strip().lower()
    if forced:
        for candidate in candidates:
            if forced in (candidate["tag"], str(candidate["backbone"]).lower()):
                return candidate
    return max(candidates, key=lambda c: (c["auc"] if c["auc"] is not None else -1.0,
                                          c["tag"]))


def model_files(key: str):
    """(modèle, métadonnées) d'une famille ; le TL est résolu dynamiquement."""
    spec = MODELS[key]
    if key == "tl":
        choice = resolve_tl()
        if choice is None:
            raise FileNotFoundError(
                "Aucun modèle Transfer Learning : lancer CNN_Transfer_Learning.ipynb "
                "(best_tl_<backbone>.keras + _meta.json).")
        return Path(choice["model_path"]), Path(choice["meta_path"])
    return Path(spec["model_path"]), Path(spec["meta_path"])


def model_label(key: str) -> str:
    spec = MODELS[key]
    if key == "tl":
        choice = resolve_tl()
        if choice:
            return f"Transfer Learning ({choice['backbone']})"
    return spec["label"]


def model_run_name(key: str) -> str:
    spec = MODELS[key]
    if key == "tl":
        choice = resolve_tl()
        return f"tl_{choice['tag']}_3fold" if choice else "tl_3fold"
    return spec["run_name"]


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
    m = (m - m.mean()) / (m.std() + 1e-6)
    # Le CNN TensorFlow est entraîné sur une version réduite (CPU) : on applique
    # exactement le même sous-échantillonnage qu'au moment du cache d'entraînement.
    downsample = int(c.get("downsample", 1))
    if downsample > 1:
        m = m[::downsample, ::downsample]
    return m.astype(np.float32)


def mel_batch(segments, cfg: Optional[dict] = None) -> np.ndarray:
    """Lot ``(B, H, W, 1)`` channels_last pour le CNN et l'hybride Transformer-CNN."""
    return np.stack([mel_spectrogram(s, cfg) for s in segments])[..., None].astype(np.float32)


def _tl_preprocess_input(backbone: str):
    """``preprocess_input`` du backbone (ResNet : mode caffe, VGG : mode caffe aussi)."""
    from tensorflow import keras

    if str(backbone or "").lower().startswith("vgg"):
        return keras.applications.vgg16.preprocess_input
    return keras.applications.resnet.preprocess_input


def transfer_learning_batch(segments, cfg: Optional[dict] = None, img_size: int = 128,
                            backbone: str = "VGG16") -> np.ndarray:
    """Pipeline exact du notebook ``CNN_Transfer_Learning``.

    Mel 64×128 → ``tf.image.resize`` 128×128 → gris→RGB → ``preprocess_input(x * 255)``.
    """
    import tensorflow as tf

    mels = np.stack([mel_spectrogram(s, cfg) for s in segments])[..., None]
    x = tf.convert_to_tensor(mels, dtype=tf.float32)
    x = tf.image.resize(x, (int(img_size), int(img_size)))
    x = tf.image.grayscale_to_rgb(x)
    x = _tl_preprocess_input(backbone)(x * 255.0)
    return np.asarray(x, dtype=np.float32)


def _hybrid_custom_objects() -> dict:
    """Reconstruit ``PositionalEncoding`` (couche personnalisée du notebook hybride)."""
    global _hybrid_layer_cls
    if _hybrid_layer_cls is not None:
        return {"PositionalEncoding": _hybrid_layer_cls}

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    @keras.utils.register_keras_serializable(package="hybrid")
    class PositionalEncoding(layers.Layer):
        """Encodage positionnel sinusoïdal — identique à Transformer_CNN_Hybrid.ipynb."""

        def __init__(self, max_len=256, d_model=128, **kwargs):
            super().__init__(**kwargs)
            self.max_len = int(max_len)
            self.d_model = int(d_model)

        def build(self, input_shape):
            pos = np.arange(self.max_len)[:, np.newaxis]
            i = np.arange(self.d_model)[np.newaxis, :]
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
            angles = pos * angle_rates
            pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
            pe[:, 0::2] = np.sin(angles[:, 0::2])
            pe[:, 1::2] = np.cos(angles[:, 1::2])
            self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
            super().build(input_shape)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            return x + tf.cast(self.pe[:, :seq_len, :], x.dtype)

        def get_config(self):
            config = super().get_config()
            config.update({"max_len": self.max_len, "d_model": self.d_model})
            return config

    _hybrid_layer_cls = PositionalEncoding
    return {"PositionalEncoding": PositionalEncoding}



# ------------------------------------------------------ découverte MLflow
def mlflow_runs(force: bool = False, ttl_s: float = 60.0) -> dict:
    """Dernier run de chaque famille dans l'expérience partagée (cache 60 s)."""
    if not force and time.time() - _mlflow_cache["at"] < ttl_s:
        return _mlflow_cache["runs"]

    _preload_tensorflow()
    runs: Dict[str, dict] = {}
    # Le serveur MLflow n'écoute parfois que sur une des deux piles (IPv4/IPv6) :
    # on essaie l'URI configurée puis son équivalent localhost/127.0.0.1.
    candidates = [MLFLOW_URI]
    for a, b in (("http://127.0.0.1", "http://localhost"), ("http://localhost", "http://127.0.0.1")):
        if MLFLOW_URI.startswith(a):
            candidates.append(b + MLFLOW_URI[len(a):])
    # Les notebooks du TP utilisent aussi le store SQLite local quand le serveur
    # MLflow n'est pas lancé : on le tente en dernier pour retrouver les runs.
    if _LOCAL_DB.exists():
        local_uri = f"sqlite:///{(ROOT / 'mlruns.db').as_posix()}"
        if local_uri not in candidates:
            candidates.append(local_uri)
    errors = []
    used_uri = MLFLOW_URI
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
        if exp is None:
            errors.append(f"{uri} -> expérience '{EXPERIMENT}' absente")
            continue
        try:
            found = client.search_runs([exp.experiment_id],
                                       order_by=["attributes.start_time DESC"],
                                       max_results=100)
            wanted = {key: model_run_name(key) for key in MODELS}
            for key, run_name in wanted.items():
                for run in found:
                    if run.data.tags.get("mlflow.runName") == run_name:
                        runs[run_name] = {
                            "run_id": run.info.run_id,
                            "run_name": run_name,
                            "experiment": EXPERIMENT,
                            "experiment_id": exp.experiment_id,
                            "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                            "params": dict(run.data.params),
                            "artifacts": [a.path for a in
                                          client.list_artifacts(run.info.run_id, "models")],
                        }
                        break
        except Exception as error:                 # métadonnées indisponibles : mode local
            errors.append(f"{uri} -> {type(error).__name__}: {error}")
            continue
        used_uri = uri
        break
    else:
        runs = {"_error": {"error": " | ".join(errors)}}

    _mlflow_cache.update(at=time.time(), runs=runs, uri=used_uri)
    return runs


def restore_from_mlflow(key: str) -> bool:
    """Récupère les artefacts locaux manquants depuis le dernier run MLflow.

    Les notebooks sauvegardent ``best_<famille>.<ext>`` et ``best_<famille>_meta.json``
    dans ``models/`` **et** les joignent au run MLflow. Si le dossier local a été
    nettoyé (ou restauré depuis un clone sans les artefacts), on télécharge ces
    fichiers depuis MLflow au lieu de relancer l'entraînement.

    Renvoie ``True`` si le fichier modèle a été restauré.
    """
    if key == "tl":
        return False                       # résolu dynamiquement parmi les best_tl_*
    try:
        model_path, meta_path = model_files(key)
    except FileNotFoundError:
        return False
    if model_path.exists():
        return False

    runs = mlflow_runs()
    run = runs.get(model_run_name(key)) or {}
    if not run.get("run_id"):
        return False

    names = [model_path.name]
    if not Path(meta_path).exists():
        names.append(Path(meta_path).name)

    try:
        import shutil
        import tempfile

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(_mlflow_cache.get("uri") or MLFLOW_URI)
        client = MlflowClient()
        restored = False
        with tempfile.TemporaryDirectory(prefix="audio-engine-mlflow-") as tmp:
            for name in names:
                try:
                    source = client.download_artifacts(run["run_id"], f"models/{name}", tmp)
                except Exception:            # artefact absent de ce run
                    continue
                if Path(source).is_file():
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, MODEL_DIR / name)
                    restored = True
        if restored:
            print(f"[audio-engine] {model_path.name} restauré depuis MLflow "
                  f"(run {run.get('run_name')} / {run['run_id']})")
        return restored
    except Exception as error:
        print(f"[audio-engine] restauration MLflow impossible pour {key} : "
              f"{type(error).__name__}: {error}")
        return False


# ------------------------------------------------------------- chargement
def load_model(key: str):
    """Charge (et met en cache) le modèle d'une famille : svm, tree, cnn, tl ou hybrid.

    Le cache est invalidé si l'artefact a été réécrit (notebook relancé pendant que le
    service tourne) : ``/predict_all`` sert donc toujours les derniers poids entraînés.
    """
    if key not in MODELS:
        raise KeyError(f"Famille inconnue : {key} (attendu : {', '.join(MODELS)})")

    spec = MODELS[key]
    path, meta_path = model_files(key)
    restore_from_mlflow(key)                 # récupère l'artefact local s'il manque
    signature = (str(path), path.stat().st_mtime if path.exists() else None)
    cached = _cache.get(key)
    if cached and cached.get("signature") == signature:
        return cached

    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} absent : lancer le notebook « {model_label(key)} » pour l'entraîner "
            f"(la section MLflow produit l'artefact).")

    meta = _read_meta_file(meta_path)
    if spec["kind"] == "keras":
        import tensorflow as tf

        # compile=False : l'inférence n'a besoin que des poids et de l'architecture.
        custom_objects = _hybrid_custom_objects() if key == "hybrid" else None
        model = tf.keras.models.load_model(path, compile=False,
                                           custom_objects=custom_objects)
        artifact = {"model": model, "meta": meta, "device": "cpu"}
    else:
        import joblib

        artifact = {"model": joblib.load(path), "meta": meta}
    artifact["signature"] = signature
    artifact["path"] = path
    artifact["meta_path"] = meta_path
    _cache[key] = artifact
    return artifact


def status() -> dict:
    """Disponibilité de chaque famille (sans charger les poids)."""
    runs = mlflow_runs()
    out = {}
    for key, spec in MODELS.items():
        try:
            model_path, meta_path = model_files(key)
        except FileNotFoundError:
            model_path, meta_path = Path(""), Path("")
        meta = _read_meta_file(meta_path) if str(meta_path) else {}
        run = runs.get(model_run_name(key), {})
        exists = bool(str(model_path)) and model_path.exists()
        metrics = run.get("metrics") or {}
        if not metrics:
            reference = meta.get("mean_metrics") or ((meta.get("metrics") or {}).get("test") or {})
            metrics = {k: reference.get(k) for k in
                       ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1",
                        "roc_auc", "auc") if isinstance(reference.get(k), (int, float))}
        out[key] = {
            "family": spec["family"],
            "label": model_label(key),
            "kind": spec["kind"],
            "available": exists,
            "model_file": model_path.name if exists else None,
            "size_mb": round(model_path.stat().st_size / 1e6, 3) if exists else None,
            "threshold": float(meta.get("threshold", spec["decision"]["threshold"])),
            "backbone": meta.get("backbone"),
            "architecture": meta.get("architecture"),
            "pretrained": meta.get("pretrained"),
            "best_fold": meta.get("best_fold"),
            "mlflow_run_id": run.get("run_id"),
            "mlflow_run_name": run.get("run_name"),
            "experiment": spec["experiment"],
            "metrics": {k: metrics.get(k) for k in
                        ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1",
                         "roc_auc", "auc")},
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
    elif key in ("cnn", "hybrid"):
        mel_cfg = meta.get("mel", DEFAULT_MEL)
        # (B, H, W, 1) : Keras utilise channels_last.
        inputs = mel_batch(segments, mel_cfg)
        probs = artifact["model"].predict(inputs, verbose=0)[:, 1]
    elif key == "tl":
        mel_cfg = meta.get("mel", DEFAULT_MEL)
        inputs = transfer_learning_batch(segments, mel_cfg,
                                         img_size=int(meta.get("img_size", 128)),
                                         backbone=meta.get("backbone") or "VGG16")
        probs = artifact["model"].predict(inputs, verbose=0)[:, 1]
    else:                                            # pragma: no cover
        raise KeyError(key)

    probs = np.asarray(probs, dtype=float)
    info = {
        "threshold": threshold,
        "decision": spec["decision"],
        "backbone": meta.get("backbone"),
        "n_features": int(len(meta.get("features", {}).get("names", []))) or None,
    }
    return probs, threshold, info



def analyze(data: bytes, filename: str = "", families: Optional[List[str]] = None,
            pre: Optional[Preprocessor] = None) -> dict:
    """Analyse multi-modèles d'un WAV : rapport JSON consommé par l'interface web."""
    pre = pre or Preprocessor()
    signal = pre.load_clean(data)
    segments = pre.segment(signal)
    starts = pre.segment_positions(signal)
    duration = len(signal) / pre.sr
    window = pre.seg_len / pre.sr

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
        run = runs.get(model_run_name(key), {})
        row = {"family": spec["family"], "key": key, "label": model_label(key),
               "experiment": spec["experiment"], "mlflow_run_id": run.get("run_id"),
               "mlflow_run_name": run.get("run_name"), "status": "ok", "error": None}
        try:
            probs, threshold, info = predict_segments(key, segments, pre)
            p_mean = float(probs.mean())
            artifact = load_model(key)
            meta = artifact["meta"] or {}
            reference = meta.get("mean_metrics") or (meta.get("metrics", {}) or {}).get("test") or {}
            row.update({
                "threshold": threshold,
                "prob_abnormal_mean": p_mean,
                "prob_abnormal_max": float(probs.max()),
                "prob_abnormal_min": float(probs.min()),
                "prediction": "abnormal" if p_mean >= threshold else "normal",
                "n_segments": int(len(probs)),
                "segment_probs": [round(float(p), 6) for p in probs],
                "backbone": meta.get("backbone"),
                "architecture": meta.get("architecture"),
                "model_file": Path(artifact["path"]).name if artifact.get("path") else None,
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
        "duration_s": round(duration, 3),
        "sample_rate_used": pre.sr,
        "n_segments": len(segments),
        "feature_summary": summary,
        "preprocessing": {
            "sample_rate": pre.sr,
            "segment_sec": round(window, 4),
            "hop_sec": round(pre.hop_len / pre.sr, 4),
            "max_segments": pre.max_seg,
            "bandpass_hz": list(pre.bandpass_hz),
            "n_fft": pre.n_fft,
            "hop_length": pre.hop,
            "n_mels": pre.n_mels,
            "fmin_hz": 20.0,
            "fmax_hz": pre.sr / 2.0,
        },
        "segments": [
            {"index": i, "start_s": round(start / pre.sr, 4),
             "end_s": round(min(duration, (start + pre.seg_len) / pre.sr), 4),
             "window_s": round(window, 4)}
            for i, start in enumerate(starts)
        ],
        "mlflow_tracking_uri_used": MLFLOW_URI,
        "experiment": EXPERIMENT,
        "models_discovered_from_mlflow": [
            {"family": k, "experiment": EXPERIMENT, "run_id": v.get("run_id"),
             "run_name": v.get("run_name"), "metrics": v.get("metrics", {})}
            for k, v in runs.items() if k in {model_run_name(key) for key in MODELS}
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
                        help="sous-ensemble : svm,tree,cnn,tl,hybrid (défaut : les cinq)")
    args = parser.parse_args()

    path = Path(args.audio)
    families = [f.strip() for f in args.families.split(",")] if args.families else None
    report = analyze(path.read_bytes(), path.name, families)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=" * 74)
    print("ANALYSE PCG — SVM / arbre / CNN / transfer learning / Transformer-CNN")
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
