"""Validation des configurations YAML QLoRA."""

from __future__ import annotations

import copy

import pytest

from llm_finetuning.src.config import (
    ConfigError,
    config_for_mlflow,
    list_configs,
    load_config,
    validate_config,
)


def test_configs_are_listed():
    configs = list_configs()
    assert configs, "Aucune configuration trouvée"
    names = {path.name for path in configs}
    assert "qlora_qwen25_7b.yaml" in names
    assert "qlora_qwen25_1_5b.yaml" in names


@pytest.mark.parametrize("path", list_configs(), ids=lambda path: path.name)
def test_config_loads_and_resolves_paths(path):
    config = load_config(path)
    assert config["model"]["name"]
    assert config["lora"]["target_modules"]
    assert config["training"]["output_dir"]
    for key in ("train", "validation", "test"):
        resolved = config["data"][key]
        assert resolved.endswith(".jsonl")
        assert "llm_finetuning" in resolved


def test_primary_config_is_7b_qlora():
    config = load_config("llm_finetuning/configs/qlora_qwen25_7b.yaml")
    assert config["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["bnb_4bit_quant_type"] == "nf4"
    assert config["training"]["gradient_accumulation_steps"] >= 1


def test_config_for_mlflow_is_flat():
    config = load_config("llm_finetuning/configs/qlora_qwen25_7b.yaml")
    flat = config_for_mlflow(config)
    assert "model.name" in flat
    assert "lora.target_modules" in flat
    assert isinstance(flat["lora.target_modules"], str)
    assert not any(isinstance(value, (dict, list)) for value in flat.values())


def test_missing_section_rejected():
    config = load_config("llm_finetuning/configs/qlora_qwen25_7b.yaml")
    broken = copy.deepcopy(config)
    del broken["lora"]
    with pytest.raises(ConfigError):
        validate_config(broken)


def test_unknown_model_rejected():
    config = load_config("llm_finetuning/configs/qlora_qwen25_7b.yaml")
    broken = copy.deepcopy(config)
    broken["model"]["name"] = "unknown/model"
    with pytest.raises(ConfigError):
        validate_config(broken)


def test_invalid_lora_rank_rejected():
    config = load_config("llm_finetuning/configs/qlora_qwen25_7b.yaml")
    broken = copy.deepcopy(config)
    broken["lora"]["r"] = 0
    with pytest.raises(ConfigError):
        validate_config(broken)


def test_local_config_fits_4gb_gpu():
    config = load_config("llm_finetuning/configs/qlora_qwen25_1_5b_local.yaml")
    assert config["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["max_seq_length"] <= 1024
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] >= 8
    assert config["training"]["gradient_checkpointing"] is True
    assert "local" in config["training"]["output_dir"]
