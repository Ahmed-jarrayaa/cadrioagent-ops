"""Contrôles de portabilité du notebook Colab/local."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "llm_finetuning" / "notebooks" / "colab_finetune.ipynb"


def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources() -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook_is_valid_json():
    data = notebook()
    assert data["nbformat"] == 4
    assert data["cells"]
    assert data["metadata"]["accelerator"] == "GPU"


def test_no_absolute_windows_paths():
    for source in sources():
        assert "C:\\Users" not in source
        assert "Data Mining\\llm_finetuning" not in source


def test_no_hardcoded_colab_chdir():
    for source in sources():
        assert "%cd /content" not in source


def test_uses_repo_relative_module_paths():
    joined = "\n".join(sources())
    assert "llm_finetuning.src.train" in joined
    assert "--config {CONFIG}" in joined
    assert "--adapter {ADAPTER}" in joined


def test_detects_repo_root():
    joined = "\n".join(sources())
    assert "find_repo_root" in joined
    assert "os.chdir(REPO_ROOT)" in joined


def test_notebook_size_is_reasonable():
    # Les sorties d'exécution locales ne sont pas versionnées, mais peuvent réapparaître
    # si le notebook est relancé : on borne seulement la taille du fichier.
    assert NOTEBOOK.stat().st_size < 2 * 1024 * 1024
