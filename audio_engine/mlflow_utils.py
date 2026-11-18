"""Suivi MLflow partagé par les notebooks du TP3 (SVM et arbre de décision).

Le serveur MLflow du projet tourne par défaut sur http://localhost:5000
(`mlflow server --backend-store-uri sqlite:///mlflow.db`). Si le serveur est
injoignable (TP hors ligne), on bascule sur un stockage SQLite local
`audio_engine/mlruns.db`, comme le fait déjà `src/train_cnn.py`.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import mlflow
import sklearn
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FALLBACK_TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns.db').as_posix()}"
DEFAULT_EXPERIMENT = "tp3_audio_engine"


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Configure l'URI de tracking (serveur, sinon SQLite local) et l'expérience."""
    uri = DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment)
    except Exception as error:                       # serveur arrêté / réseau coupé
        print(f"[MLflow] {uri} injoignable ({error}) -> bascule sur {FALLBACK_TRACKING_URI}")
        uri = FALLBACK_TRACKING_URI
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
    return uri


def _flatten(prefix, value, out):
    """MLflow n'accepte que des paramètres scalaires : on aplatit et on sérialise le reste."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    elif isinstance(value, (list, tuple, set)):
        _flatten(prefix, ",".join(str(v) for v in value)[:5000], out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix[:250]] = str(value) if value is not None else "None"
    else:
        out[prefix[:250]] = str(value)[:5000]


def log_classifier_run(
    *,
    run_name: str,
    model,
    params: dict,
    mean_metrics: dict,
    fold_metrics: list | None = None,
    artifacts: dict | None = None,
    input_example=None,
    tags: dict | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    model_artifact: str = "model",
) -> tuple[str, str]:
    """Enregistre un run complet (params, métriques, artefacts, modèle) dans MLflow.

    Args:
        run_name: nom lisible du run dans l'interface MLflow.
        model: estimateur scikit-learn entraîné (Pipeline accepté).
        params: hyperparamètres / configuration (dict aplati automatiquement).
        mean_metrics: métriques moyennes sur les plis.
        fold_metrics: liste de dicts, un par pli (les clés deviennent `foldK_<métrique>`).
        artifacts: {chemin local: sous-dossier d'artefacts} pour les fichiers à joindre.
        input_example: exemple d'entrée pour la signature du modèle.
        tags: tags MLflow additionnels.

    Returns:
        (tracking_uri, run_id)
    """
    uri = setup_mlflow(experiment)

    flat_params = {}
    _flatten("", params, flat_params)
    flat_params.update({
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "platform": platform.platform()[:250],
    })

    metrics = {str(k): float(v) for k, v in mean_metrics.items()}
    for row in fold_metrics or []:
        fold = row.get("fold", "?")
        for key, value in row.items():
            if key != "fold" and isinstance(value, (int, float)):
                metrics[f"fold{fold}_{key}"] = float(value)

    signature = None
    if input_example is not None:
        signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": "tp3_audio_engine",
            "model_family": type(model).__name__,
            "dataset": "ourdataset/train (normal/abnormal)",
            "features": "preprocess.Preprocessor (46 features / segment 5 s)",
            **(tags or {}),
        })
        mlflow.log_params(flat_params)
        mlflow.log_metrics(metrics)

        for local_path, artifact_dir in (artifacts or {}).items():
            path = Path(local_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=artifact_dir)
            else:
                print(f"[MLflow] artefact absent, ignoré : {path}")

        mlflow.sklearn.log_model(
            model,
            artifact_path=model_artifact,
            signature=signature,
            input_example=input_example,
        )
        run_id = run.info.run_id

    experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
    print(f"[MLflow] run '{run_name}' enregistré : {uri} / {experiment} (id={experiment_id})")
    if uri.startswith("http"):
        print(f"[MLflow] {uri}/#/experiments/{experiment_id}/runs/{run_id}")
    print(f"[MLflow] run_id = {run_id}")
    return uri, run_id


def log_torch_run(
    *,
    run_name: str,
    model,
    params: dict,
    mean_metrics: dict,
    artifacts: dict | None = None,
    input_example=None,
    tags: dict | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    model_artifact: str = "model",
) -> tuple[str, str]:
    """Équivalent de :func:`log_classifier_run` pour un ``torch.nn.Module``.

    Le modèle est sérialisé par ``mlflow.pytorch`` (rechargeable avec
    ``mlflow.pytorch.load_model``) ; les poids bruts et les métadonnées JSON
    sont jointes en artefacts pour le service d'inférence.
    """
    import torch
    from mlflow.tracking import MlflowClient

    uri = setup_mlflow(experiment)

    flat_params = {}
    _flatten("", params, flat_params)
    flat_params.update({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform()[:250],
    })
    metrics = {str(k): float(v) for k, v in mean_metrics.items()}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": "tp3_audio_engine",
            "model_family": type(model).__name__,
            "dataset": "ourdataset/train (normal/abnormal)",
            "features": "Mel 128x256 (preprocess.Preprocessor, segments 5 s)",
            **(tags or {}),
        })
        mlflow.log_params(flat_params)
        mlflow.log_metrics(metrics)

        for local_path, artifact_dir in (artifacts or {}).items():
            path = Path(local_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=artifact_dir)
            else:
                print(f"[MLflow] artefact absent, ignoré : {path}")

        model.eval()
        mlflow.pytorch.log_model(model, artifact_path=model_artifact, input_example=input_example)
        run_id = run.info.run_id

    experiment_id = MlflowClient().get_experiment_by_name(experiment).experiment_id
    print(f"[MLflow] run '{run_name}' enregistré : {uri} / {experiment} (id={experiment_id})")
    if uri.startswith("http"):
        print(f"[MLflow] {uri}/#/experiments/{experiment_id}/runs/{run_id}")
    print(f"[MLflow] run_id = {run_id}")
    return uri, run_id
