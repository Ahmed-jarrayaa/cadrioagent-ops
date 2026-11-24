# Vérifications locales (sans GPU, sans téléchargement de modèle).
# Usage : .\llm_finetuning\scripts\run_local_checks.ps1
$ErrorActionPreference = "Stop"

Write-Host "== Construction du dataset =="
python llm_finetuning\data\verify_sources.py
python llm_finetuning\data\build_dataset.py
python llm_finetuning\data\build_dataset.py --check

Write-Host "== Tests =="
python -m pytest llm_finetuning\tests -q

Write-Host "== Dry-run de la configuration 7B =="
python -m llm_finetuning.src.train --config llm_finetuning\configs\qlora_qwen25_7b.yaml --dry-run

Write-Host "== OK =="
