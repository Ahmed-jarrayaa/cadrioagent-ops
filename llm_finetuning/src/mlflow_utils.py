"""Suivi MLflow de la phase LLM Fine-Tuning.

Même logique que ``audio_engine/mlflow_utils.py`` : on tente le serveur MLflow
(configuré par ``MLFLOW_TRACKING_URI`` ou la config), puis on bascule sur une
base SQLite locale si le serveur est injoignable. Cela permet d'exécuter
l'entraînement sur Google Colab sans serveur distant.
"""

from __future__ import annotations

import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_DB = REPO_ROOT / "llm_finetuning" / "mlruns.db"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_EXPERIMENT = "cardioagent_llm"

_METRIC_KEYS = ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch")


def setup_mlflow(
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    """Configure MLflow et renvoie l'URI réellement utilisée."""
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment)
    except Exception as error:  # serveur arrêté, réseau, permissions
        fallback = f"sqlite:///{(FALLBACK_DB).as_posix()}"
        print(f"[MLflow] {uri} injoignable ({type(error).__name__}: {error}) -> repli sur {fallback}")
        uri = fallback
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
    return uri


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def log_params_safe(params: dict[str, Any]) -> None:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        value = _scalar(value)
        if value is None:
            value = "None"
        text = str(value)
        if len(text) > 5000:
            text = text[:5000]
        flat[key[:250]] = text
    mlflow.log_params(flat)


def log_environment() -> None:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "unsloth", "mlflow"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "inconnue")
        except Exception:
            versions[module_name] = "absent"
    try:
        import torch

        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        else:
            versions["gpu"] = "cpu"
    except Exception:
        versions["gpu"] = "inconnu"
    log_params_safe({f"env.{key}": value for key, value in versions.items()})


def log_dataset_manifest(
    *,
    fingerprints: dict[str, str],
    stats: dict[str, Any],
    sources_file: str | Path | None = None,
) -> None:
    payload = {"fingerprints": fingerprints, "stats": stats}
    mlflow.log_dict(payload, "dataset/manifest.json")
    if sources_file is not None:
        path = Path(sources_file)
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="dataset")


def log_evaluation(metrics: dict[str, float], report: str | None = None) -> None:
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if numeric:
        mlflow.log_metrics(numeric)
    if report:
        mlflow.log_text(report, "evaluation/evaluation_report.md")


def log_artifacts(*paths: str | Path, artifact_path: str = "artifacts") -> None:
    for path in paths:
        path = Path(path)
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        else:
            print(f"[MLflow] artefact absent, ignoré : {path}")


def _make_callback():
    from transformers import TrainerCallback

    class MlflowMetricsCallback(TrainerCallback):
        """Enregistre les pertes et métriques d'entraînement pas à pas."""

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if not logs:
                return
            step = int(state.global_step)
            for key in _METRIC_KEYS:
                value = logs.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    mlflow.log_metric(key, float(value), step=step)

    return MlflowMetricsCallback()


def trainer_callbacks() -> list:
    """Callbacks MLflow à passer au ``Trainer``."""
    return [_make_callback()]


def print_run_url(experiment: str) -> None:
    client = MlflowClient()
    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        return
    uri = mlflow.get_tracking_uri()
    if str(uri).startswith("http"):
        print(f"[MLflow] {uri}/#/experiments/{experiment_obj.experiment_id}")
