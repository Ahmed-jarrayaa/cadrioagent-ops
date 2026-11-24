"""Validation des artefacts d'export GGUF / Ollama."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = REPO_ROOT / "llm_finetuning"


def test_modelfile_template():
    template = (LLM_DIR / "export" / "Modelfile.template").read_text(encoding="utf-8")
    assert "FROM __GGUF_FILE__" in template
    assert "TEMPLATE" in template
    assert "SYSTEM" in template
    assert "PARAMETER num_ctx" in template
    assert "diagnostic" in template.lower()


def test_manifest_example_is_valid():
    manifest = json.loads((LLM_DIR / "export" / "MANIFEST.example.json").read_text(encoding="utf-8"))
    for key in ("base_model", "quantization", "gguf_file", "gguf_sha256", "dataset_version", "disclaimer"):
        assert key in manifest
    assert manifest["quantization"] == "q4_k_m"


def test_gitignore_covers_large_artifacts():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.gguf" in text
    assert "llm_finetuning/outputs/" in text
    assert "llm_finetuning/data/raw/" in text


def test_readme_documents_key_steps():
    readme = (LLM_DIR / "README.md").read_text(encoding="utf-8")
    for keyword in ("QLoRA", "Colab", "MLflow", "GGUF", "Ollama", "licence"):
        assert keyword.lower() in readme.lower(), keyword
