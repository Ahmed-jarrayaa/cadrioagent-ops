# Entraînement local QLoRA 1.5B sur GPU 4 Go (Windows).
# Usage :
#   .\llm_finetuning\scripts\train_local.ps1
#   .\llm_finetuning\scripts\train_local.ps1 -MaxSteps 20     # smoke test
param(
    [int]$MaxSteps = -1
)
$ErrorActionPreference = "Stop"

$config = "llm_finetuning\configs\qlora_qwen25_1_5b_local.yaml"

Write-Host "== Vérification CUDA =="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible : installer torch cu121'; print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'Go')"

Write-Host "== Validation de la configuration et du dataset =="
python -m llm_finetuning.src.train --config $config --dry-run

# Réduit la fragmentation mémoire sur les petits GPU.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if ($MaxSteps -gt 0) {
    Write-Host "== Smoke test : $MaxSteps pas =="
    python -m llm_finetuning.src.train --config $config --max-steps $MaxSteps
} else {
    Write-Host "== Entraînement complet =="
    python -m llm_finetuning.src.train --config $config
}
