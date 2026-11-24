#!/usr/bin/env bash
# Exécution de bout en bout sur Google Colab / Linux GPU.
# Usage : bash llm_finetuning/scripts/run_colab.sh
set -euo pipefail

CONFIG="${CONFIG:-llm_finetuning/configs/qlora_qwen25_7b.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-llm_finetuning/outputs/qwen25-7b-cardio/adapter}"
QUANT="${QUANT:-q4_k_m}"

echo "== 1. Installation des dépendances =="
python -m pip install -U pip
pip install -r llm_finetuning/requirements-llm.txt

echo "== 2. Construction et validation du dataset =="
python llm_finetuning/data/build_dataset.py
python llm_finetuning/data/build_dataset.py --check
pytest llm_finetuning/tests -q

echo "== 3. Vérification à sec de la configuration =="
python -m llm_finetuning.src.train --config "$CONFIG" --dry-run

echo "== 4. Entraînement QLoRA =="
python -m llm_finetuning.src.train --config "$CONFIG"

echo "== 5. Évaluation indépendante =="
python -m llm_finetuning.src.evaluate \
  --config "$CONFIG" \
  --adapter "$ADAPTER_DIR" \
  --compare-base

echo "== 6. Export GGUF =="
if [ ! -d llm_finetuning/export/llama.cpp ]; then
  git clone https://github.com/ggerganov/llama.cpp llm_finetuning/export/llama.cpp
  cmake -S llm_finetuning/export/llama.cpp -B llm_finetuning/export/llama.cpp/build
  cmake --build llm_finetuning/export/llama.cpp/build --config Release -j
fi
python -m llm_finetuning.src.export_gguf \
  --config "$CONFIG" \
  --adapter "$ADAPTER_DIR" \
  --output-dir llm_finetuning/export \
  --quantization "$QUANT"

echo "== Terminé. Télécharger llm_finetuning/export/ et llm_finetuning/outputs/ =="
