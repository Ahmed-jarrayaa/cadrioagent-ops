"""Chargement et validation des configurations YAML du fine-tuning LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "llm_finetuning" / "configs"

REQUIRED_SECTIONS = ("model", "lora", "training", "data", "mlflow")
REQUIRED_MODEL_FIELDS = ("name", "load_in_4bit", "max_seq_length")
REQUIRED_LORA_FIELDS = ("r", "alpha", "target_modules")
REQUIRED_TRAINING_FIELDS = (
    "output_dir",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
)
REQUIRED_DATA_FIELDS = ("train", "validation", "test")
ALLOWED_MODEL_PREFIXES = ("Qwen/", "meta-llama/", "unsloth/")


class ConfigError(ValueError):
    """Configuration invalide."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Charge, valide et normalise une configuration YAML."""
    config_path = resolve_path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration introuvable : {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"{config_path} ne contient pas un objet YAML.")

    validate_config(config, config_path)
    config["_config_path"] = str(config_path)
    for key in REQUIRED_DATA_FIELDS:
        config["data"][key] = str(resolve_path(config["data"][key]))
    config["training"]["output_dir"] = str(resolve_path(config["training"]["output_dir"]))
    return config


def _require(container: dict, fields: tuple[str, ...], section: str) -> None:
    missing = [field for field in fields if field not in container]
    if missing:
        raise ConfigError(f"Section '{section}' : champs manquants {missing}")


def validate_config(config: dict, source: str | Path = "<config>") -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"{source} : sections manquantes {missing}")

    model = config["model"]
    lora = config["lora"]
    training = config["training"]
    data = config["data"]
    mlflow = config["mlflow"]

    _require(model, REQUIRED_MODEL_FIELDS, "model")
    _require(lora, REQUIRED_LORA_FIELDS, "lora")
    _require(training, REQUIRED_TRAINING_FIELDS, "training")
    _require(data, REQUIRED_DATA_FIELDS, "data")

    if not isinstance(model["name"], str) or not model["name"].startswith(ALLOWED_MODEL_PREFIXES):
        raise ConfigError(
            f"{source} : modèle '{model['name']}' non autorisé. "
            f"Préfixes acceptés : {ALLOWED_MODEL_PREFIXES}."
        )
    if not 1 <= int(lora["r"]) <= 128:
        raise ConfigError("lora.r doit être compris entre 1 et 128.")
    if int(lora["alpha"]) < int(lora["r"]):
        raise ConfigError("lora.alpha doit être supérieur ou égal à lora.r.")
    if not 0 <= float(lora["dropout"]) < 1:
        raise ConfigError("lora.dropout doit être dans [0, 1).")
    if not isinstance(lora["target_modules"], list) or not lora["target_modules"]:
        raise ConfigError("lora.target_modules doit être une liste non vide.")
    if not 128 <= int(model["max_seq_length"]) <= 32768:
        raise ConfigError("model.max_seq_length doit être compris entre 128 et 32768.")
    if float(training["learning_rate"]) <= 0:
        raise ConfigError("training.learning_rate doit être positif.")
    if int(training["per_device_train_batch_size"]) < 1:
        raise ConfigError("training.per_device_train_batch_size doit être >= 1.")
    if int(training["gradient_accumulation_steps"]) < 1:
        raise ConfigError("training.gradient_accumulation_steps doit être >= 1.")
    if int(training["num_train_epochs"]) < 1 and int(training.get("max_steps", -1)) < 1:
        raise ConfigError("Définir num_train_epochs >= 1 ou max_steps >= 1.")
    if not isinstance(mlflow.get("experiment"), str) or not mlflow["experiment"]:
        raise ConfigError("mlflow.experiment doit être une chaîne non vide.")

    for key in REQUIRED_DATA_FIELDS:
        if not str(data[key]).strip():
            raise ConfigError(f"data.{key} est vide.")


def config_for_mlflow(config: dict) -> dict[str, Any]:
    """Aplatit la configuration pour ``mlflow.log_params`` (valeurs scalaires)."""
    flat: dict[str, Any] = {}
    for section, values in config.items():
        if section.startswith("_") or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            flat[f"{section}.{key}"] = value
    return flat


def list_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))
