"""Entraînement QLoRA 4-bit (Unsloth / PEFT + TRL) avec suivi MLflow.

Exemples :
    python -m llm_finetuning.src.train --config llm_finetuning/configs/qlora_qwen25_7b.yaml --dry-run
    python -m llm_finetuning.src.train --config llm_finetuning/configs/qlora_qwen25_7b.yaml
    python -m llm_finetuning.src.train --config ... --max-steps 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlflow

from llm_finetuning.src import mlflow_utils
from llm_finetuning.src.config import config_for_mlflow, load_config
from llm_finetuning.src.data import (
    fingerprint,
    format_for_training,
    load_records,
    load_text_dataset,
    split_stats,
)


def _training_arguments(training_cfg: dict[str, Any]):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": training_cfg["output_dir"],
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "optim": str(training_cfg.get("optim", "paged_adamw_8bit")),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_strategy": str(training_cfg.get("save_strategy", "steps")),
        "save_steps": int(training_cfg.get("save_steps", 50)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(training_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "bf16": bool(training_cfg.get("bf16", False)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "seed": int(training_cfg.get("seed", 42)),
        "report_to": list(training_cfg.get("report_to", [])),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
        "remove_unused_columns": False,
    }
    # transformers >= 4.46 renomme evaluation_strategy en eval_strategy.
    try:
        return TrainingArguments(eval_strategy=str(training_cfg.get("eval_strategy", "steps")), **kwargs)
    except TypeError:
        return TrainingArguments(
            evaluation_strategy=str(training_cfg.get("eval_strategy", "steps")), **kwargs
        )


def _build_trainer(model, tokenizer, config: dict[str, Any], train_dataset, eval_dataset):
    from transformers import TrainerCallback
    from trl import SFTTrainer

    training_cfg = config["training"]
    model_cfg = config["model"]
    callbacks = list(mlflow_utils.trainer_callbacks())

    class SaveConfigCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config["_config_path"], output_dir / "training_config.yaml")

    callbacks.append(SaveConfigCallback())

    common = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "dataset_text_field": "text",
        "max_seq_length": int(model_cfg["max_seq_length"]),
        "args": _training_arguments(training_cfg),
        "callbacks": callbacks,
        "packing": False,
    }
    try:
        return SFTTrainer(**common, dataset_num_proc=1)
    except TypeError:
        common.pop("max_seq_length", None)
        return SFTTrainer(**common, dataset_num_proc=1)


def run(config_path: str, *, dry_run: bool = False, max_steps: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if max_steps is not None:
        config["training"]["max_steps"] = int(max_steps)

    data_paths = config["data"]
    stats = split_stats(data_paths)
    prints = {
        "config": config["_config_path"],
        "model": config["model"]["name"],
        "splits": stats,
    }
    print(json.dumps(prints, ensure_ascii=False, indent=2))

    if dry_run:
        records = load_records(data_paths["train"])
        print(f"[dry-run] {len(records)} exemples d'entraînement chargés et valides.")
        print("[dry-run] Aucun modèle téléchargé. Configuration et dataset OK.")
        return {"status": "dry-run", "splits": stats}

    from llm_finetuning.src.data import load_text_dataset
    from llm_finetuning.src.model import gpu_summary, load_model_and_tokenizer

    print(f"[GPU] {gpu_summary()}")

    model, tokenizer, backend = load_model_and_tokenizer(config)
    train_dataset = load_text_dataset(data_paths["train"], tokenizer)
    eval_dataset = load_text_dataset(data_paths["validation"], tokenizer)

    uri = mlflow_utils.setup_mlflow(
        experiment=config["mlflow"]["experiment"],
        tracking_uri=config["mlflow"].get("tracking_uri"),
    )
    print(f"[MLflow] tracking URI : {uri}")

    with mlflow.start_run(run_name=config["mlflow"].get("run_name")) as run_obj:
        mlflow.set_tags(
            {
                "project": "cardioagent_ops",
                "phase": "llm_finetuning",
                "method": "qlora_4bit",
                "backend": backend,
                "model_family": config["model"]["name"],
                "dataset": "cardioagent-llm-fr-en v1.0.0",
                "disclaimer": "educational_only_not_a_medical_device",
            }
        )
        params = config_for_mlflow(config)
        mlflow_utils.log_params_safe(params)
        mlflow_utils.log_environment()
        mlflow_utils.log_dataset_manifest(
            fingerprints=fingerprint(data_paths.values()),
            stats=stats,
            sources_file="llm_finetuning/data/sources.json",
        )

        trainer = _build_trainer(model, tokenizer, config, train_dataset, eval_dataset)
        train_result = trainer.train()
        metrics = {f"train_{key}": value for key, value in train_result.metrics.items()}
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        )

        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            }
        )

        adapter_dir = Path(config["training"]["output_dir"]) / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        trainer.save_state()

        mlflow_utils.log_artifacts(
            config["_config_path"],
            Path(config["training"]["output_dir"]) / "trainer_state.json",
            adapter_dir / "adapter_config.json",
            artifact_path="training",
        )
        mlflow_utils.log_artifacts(adapter_dir, artifact_path="adapter")

        run_id = run_obj.info.run_id
        print(f"[MLflow] run_id = {run_id}")
        print(f"[train] adaptateur sauvegardé : {adapter_dir}")
        mlflow_utils.print_run_url(config["mlflow"]["experiment"])

    return {
        "status": "trained",
        "run_id": run_id,
        "adapter_dir": str(adapter_dir),
        "eval_metrics": eval_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Chemin du YAML de configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Valide sans charger le modèle.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limite le nombre de pas (smoke test).")
    args = parser.parse_args()
    result = run(args.config, dry_run=args.dry_run, max_steps=args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
