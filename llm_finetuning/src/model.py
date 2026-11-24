"""Chargement du modèle Qwen et application de l'adaptateur LoRA/QLoRA.

Deux chemins sont supportés :
1. **Unsloth** (recommandé sur Colab) : chargement 4-bit optimisé et LoRA rapide.
2. **Fallback Transformers + PEFT + bitsandbytes** si Unsloth est indisponible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _torch_dtype(name: str | None):
    if name in (None, "auto"):
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"dtype inconnu : {name}")
    return mapping[name]


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "GPU : aucun (CPU)"
    props = torch.cuda.get_device_properties(0)
    return f"GPU : {props.name} — {props.total_memory / 1024**3:.1f} Go VRAM"


def load_model_and_tokenizer(config: dict[str, Any]):
    """Charge le modèle quantifié 4-bit et applique LoRA.

    Returns:
        (model, tokenizer, backend) où backend vaut ``unsloth`` ou ``peft``.
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=int(model_cfg["max_seq_length"]),
            dtype=_torch_dtype(model_cfg.get("dtype")),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg["r"]),
            target_modules=list(lora_cfg["target_modules"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=str(lora_cfg.get("bias", "none")),
            use_gradient_checkpointing=str(lora_cfg.get("use_gradient_checkpointing", "unsloth")),
            random_state=int(config["training"].get("seed", 42)),
            use_rslora=False,
            loftq_config=None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, "unsloth"
    except ImportError:
        print("[model] Unsloth indisponible -> fallback Transformers + PEFT + bitsandbytes")
    except Exception as error:
        print(f"[model] Unsloth a échoué ({type(error).__name__}: {error}) -> fallback PEFT")

    return _load_with_peft(config)


def _load_with_peft(config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    compute_dtype = _torch_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    quantization_config = None
    if bool(model_cfg.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=_torch_dtype(model_cfg.get("dtype")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True))
        )

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, "peft"


def load_for_inference(config: dict[str, Any], adapter_path: str | Path | None = None):
    """Charge le modèle (base ou adaptateur) pour l'évaluation/l'export."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    quantization_config = None
    if bool(model_cfg.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=_torch_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=_torch_dtype(model_cfg.get("dtype")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def merge_and_save(model, tokenizer, output_dir: str | Path) -> Path:
    """Fusionne l'adaptateur LoRA dans le modèle de base et sauvegarde en fp16."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        raise RuntimeError("Le modèle ne supporte pas merge_and_unload (PEFT requis).")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
