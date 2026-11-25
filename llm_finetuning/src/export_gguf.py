"""Export du modèle fine-tuné au format GGUF pour Ollama.

Deux méthodes :
1. ``--method llama_cpp`` (recommandé, reproductible) :
   - fusion de l'adaptateur LoRA dans un modèle HF fp16 ;
   - conversion via ``llama.cpp/convert_hf_to_gguf.py`` ;
   - quantification via ``llama-quantize`` (q4_k_m par défaut).
2. ``--method unsloth`` : export direct via ``save_pretrained_gguf`` (si disponible).

Le script écrit aussi un ``MANIFEST.json`` (taille, SHA-256, quantification) et peut
générer un ``Modelfile`` Ollama à partir du template.

Exemple Colab :
    python -m llm_finetuning.src.export_gguf \
        --config llm_finetuning/configs/qlora_qwen25_7b.yaml \
        --adapter llm_finetuning/outputs/qwen25-7b-cardio/adapter \
        --output-dir llm_finetuning/export \
        --quantization q4_k_m
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_finetuning.src.config import load_config, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPO_ROOT / "llm_finetuning" / "export"
TEMPLATE = DEFAULT_EXPORT_DIR / "Modelfile.template"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_adapter_to_fp16(
    config: dict, adapter: str, merged_dir: Path, device: str = "auto"
) -> Path:
    """Fusionne l'adaptateur dans le modèle de base (poids fp16).

    ``device="cpu"`` charge le modèle en float32 le temps de la fusion, puis le
    repasse en float16 avant sauvegarde. C'est indispensable sur un GPU de 4 Go :
    la fusion fp16 sur GPU sature la VRAM.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"[export] chargement de {model_name} sur {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    if device == "cpu":
        merged = merged.half()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[export] modèle fusionné : {merged_dir}")
    return merged_dir


def export_with_llama_cpp(
    merged_dir: Path,
    output_dir: Path,
    quantization: str,
    llama_cpp_dir: Path,
) -> Path:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"{convert_script} introuvable. Cloner llama.cpp dans {llama_cpp_dir} "
            "(voir README)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "cardio-llm-f16.gguf"
    print(f"[export] conversion HF -> GGUF f16 : {f16_path}")
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    quantize_bin = None
    for candidate in (
        llama_cpp_dir / "llama-quantize.exe",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
    ):
        if candidate.exists():
            quantize_bin = candidate
            break
    if quantize_bin is None:
        raise FileNotFoundError(
            "llama-quantize introuvable. Compiler llama.cpp ou placer le binaire "
            "précompilé dans le dossier llama.cpp (voir README)."
        )

    final_path = output_dir / f"cardio-llm-{quantization}.gguf"
    print(f"[export] quantification {quantization} : {final_path}")
    subprocess.run([str(quantize_bin), str(f16_path), str(final_path), quantization.upper()], check=True)
    return final_path


def export_with_unsloth(config: dict, adapter: str, output_dir: Path, quantization: str) -> Path:
    """Export direct via Unsloth (si l'adaptateur a été entraîné avec Unsloth)."""
    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=int(config["model"]["max_seq_length"]),
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(output_dir / "cardio-llm"),
        tokenizer,
        quantization_method=quantization,
    )
    candidates = sorted(output_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError("Aucun GGUF produit par Unsloth.")
    return candidates[-1]


def write_modelfile(gguf_path: Path, output_dir: Path, model_name: str = "cardio-llm") -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template absent : {TEMPLATE}")
    content = TEMPLATE.read_text(encoding="utf-8")
    content = content.replace("__GGUF_FILE__", gguf_path.name)
    content = content.replace("__MODEL_NAME__", model_name)
    path = output_dir / "Modelfile"
    path.write_text(content, encoding="utf-8")
    print(f"[export] Modelfile : {path}")
    return path


def write_manifest(
    *,
    config: dict,
    adapter: str,
    gguf_path: Path,
    quantization: str,
    output_dir: Path,
) -> Path:
    manifest = {
        "model_name": "cardio-llm",
        "base_model": config["model"]["name"],
        "base_model_revision": config["model"].get("revision", "main"),
        "adapter": str(adapter),
        "quantization": quantization,
        "gguf_file": gguf_path.name,
        "gguf_size_bytes": gguf_path.stat().st_size,
        "gguf_sha256": sha256_file(gguf_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "cardioagent-llm-fr-en v1.0.0",
        "license_note": "Vérifier la licence du modèle de base (Qwen2.5 : Apache-2.0) et des sources.",
        "disclaimer": "Modèle pédagogique. Pas un dispositif médical. Ne pas utiliser pour diagnostiquer.",
    }
    path = output_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[export] manifeste : {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--quantization", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0"])
    parser.add_argument("--method", default="llama_cpp", choices=["llama_cpp", "unsloth"])
    parser.add_argument("--llama-cpp-dir", default=str(REPO_ROOT / "llm_finetuning" / "export" / "llama.cpp"))
    parser.add_argument(
        "--merge-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Périphérique de fusion de l'adaptateur (cpu conseillé sur GPU 4 Go).",
    )
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Conserver le modèle HF fusionné (3 Go pour un 1.5B).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged"

    if args.merge_only:
        merge_adapter_to_fp16(config, args.adapter, merged_dir, args.merge_device)
        return 0

    if args.method == "unsloth":
        gguf_path = export_with_unsloth(config, args.adapter, output_dir, args.quantization)
    else:
        merge_adapter_to_fp16(config, args.adapter, merged_dir, args.merge_device)
        gguf_path = export_with_llama_cpp(
            merged_dir, output_dir, args.quantization, Path(args.llama_cpp_dir)
        )

    write_modelfile(gguf_path, output_dir)
    write_manifest(
        config=config,
        adapter=args.adapter,
        gguf_path=gguf_path,
        quantization=args.quantization,
        output_dir=output_dir,
    )

    if args.method == "llama_cpp":
        # Le f16 intermédiaire peut être supprimé pour économiser le disque.
        f16_path = output_dir / "cardio-llm-f16.gguf"
        if f16_path.exists() and f16_path != gguf_path:
            f16_path.unlink()
            print(f"[export] f16 intermédiaire supprimé : {f16_path}")
        if not args.keep_merged and merged_dir.exists():
            shutil.rmtree(merged_dir)
            print(f"[export] modèle HF fusionné supprimé : {merged_dir}")

    print(f"[export] terminé : {gguf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
