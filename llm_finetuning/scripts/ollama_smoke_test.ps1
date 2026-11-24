# Smoke test Ollama pour le modèle cardio-llm.
# Prérequis : ollama installé, GGUF + Modelfile dans llm_finetuning/export/.
$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama n'est pas installé ou n'est pas dans le PATH."
}
if (-not (Test-Path "llm_finetuning\export\Modelfile")) {
    throw "llm_finetuning\export\Modelfile absent. Lancer export_gguf.py d'abord."
}

Write-Host "== Création du modèle cardio-llm =="
ollama create cardio-llm -f llm_finetuning\export\Modelfile

$prompts = @(
    "Qu'est-ce que le premier bruit du coeur B1 ?",
    "J'ai une douleur thoracique intense, est-ce un infarctus ?",
    "What is the role of cardiac auscultation?"
)

foreach ($prompt in $prompts) {
    Write-Host "`n== Prompt : $prompt =="
    $prompt | ollama run cardio-llm
}
