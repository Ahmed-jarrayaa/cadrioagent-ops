# LLM Fine-Tuning — CardioAgent-Ops

Adaptation **QLoRA 4-bit** d'un modèle open source (Qwen2.5-7B-Instruct) sur un corpus
pédagogique de cardiologie, avec suivi **MLflow**, évaluation indépendante et export **GGUF**
pour **Ollama**.

> ⚠️ **Avertissement** : ce module est un travail **académique**. Le modèle obtenu n'est **pas un
> dispositif médical**, ne pose pas de diagnostic et ne doit jamais servir à une décision
> clinique. Il ne remplace ni l'examen physique ni un professionnel de santé.

---

## 1. Périmètre et contraintes

| Élément | Décision |
|---|---|
| Modèle principal | **Qwen2.5-7B-Instruct** (QLoRA 4-bit) |
| Exécution d'entraînement | **Google Colab** GPU T4 16 Go ou L4 24 Go |
| Fallback | **Qwen2.5-1.5B-Instruct** (smoke test / très petites ressources) |
| Langues | français + anglais |
| Sources | **uniquement les 2 PDF fournis** dans `rag_engine/` |
| Volume | **450 train / 70 validation / 70 test** |
| Suivi | MLflow (expérience `cardioagent_llm`) |
| Export | GGUF `q4_k_m` (+ option `q5_k_m`) |
| Intégration | Ollama, via `llm_finetuning/export/Modelfile` |

### Pourquoi Colab ?

L'audit de la machine locale a relevé :

- GPU **RTX 2050 Laptop, 4 Go de VRAM** → insuffisant pour un 7B même en 4-bit ;
- **16 Go de RAM** et **≈ 16 Go de disque libres** → insuffisants pour la base 7B fp16 et la
  stack d'entraînement ;
- Windows natif → support limité de `bitsandbytes`/Unsloth.

Le code local sert donc à : construire/valider le dataset, lancer les tests, documenter et
préparer l'export. L'entraînement réel se fait sur Colab.

---

## 2. Arborescence

```text
llm_finetuning/
├── README.md                     # ce document
├── requirements-llm.txt          # stack Colab/Linux
├── configs/
│   ├── qlora_qwen25_7b.yaml      # configuration principale
│   └── qlora_qwen25_1_5b.yaml    # repli / smoke test
├── data/
│   ├── sources.json              # registre des sources et licences
│   ├── seed_facts.jsonl          # 90 fiches de faits ancrées (FR/EN)
│   ├── seed_guardrails.jsonl     # 50 consignes de sécurité (FR/EN)
│   ├── build_dataset.py          # génération + découpage + contrôles
│   ├── raw/                      # texte extrait des PDF (non versionné)
│   └── dataset/
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       ├── dataset_card.md
│       ├── dataset_stats.json
│       └── split_manifest.json
├── src/
│   ├── config.py                 # chargement/validation YAML
│   ├── data.py                   # lecture JSONL + template ChatML
│   ├── model.py                  # Qwen + LoRA/QLoRA (Unsloth/PEFT)
│   ├── train.py                  # entraînement + MLflow
│   ├── evaluate.py               # évaluation indépendante
│   ├── export_gguf.py            # fusion + conversion + quantification
│   └── mlflow_utils.py           # tracking MLflow
├── notebooks/
│   └── colab_finetune.ipynb      # notebook Colab (wrapper)
├── scripts/
│   ├── run_colab.sh
│   ├── run_local_checks.ps1
│   └── ollama_smoke_test.ps1
├── export/
│   ├── Modelfile.template
│   ├── MANIFEST.example.json
│   └── .gitkeep
└── tests/                        # pytest, sans GPU
```

---

## 3. Dataset

### 3.1 Sources et licences

| ID | Document | Licence / droits |
|---|---|---|
| `auscultation_cardiaque_normale` | Cours « Auscultation cardiaque normale », Dr F. Touati, Université Salah Boubenider Constantine 3, 2024-2025 | Droits non explicités → **usage académique interne**, pas de redistribution du texte intégral |
| `esc_evaluation_cardiovasculaire` | Huette P., Abou-Arab O. « Les nouvelles recommandations de la Société européenne de cardiologie… », 2023 | **CC BY-NC 4.0** → usage non commercial avec attribution |

Le registre complet est dans [`data/sources.json`](data/sources.json), avec l'empreinte SHA-256 de
chaque PDF et le chemin du texte extrait. Les textes bruts extraits (`data/raw/`) ne sont **pas
versionnés** : ils peuvent être régénérés localement.

Vérifier la provenance (PDF → empreinte → texte) :

```bash
python llm_finetuning/data/verify_sources.py
```

Sortie attendue : `[OK] Provenance des sources vérifiée.` Le script échoue si un PDF a été
remplacé ou si le texte extrait ne correspond plus au document.

Sur une machine où `data/raw/` n'existe pas encore (clone neuf, Colab), régénérer le texte :

```bash
python llm_finetuning/data/verify_sources.py --refresh
```

### 3.2 Format

Chaque ligne JSONL contient :

```json
{
  "id": "ausc_01::qa_fr",
  "instruction": "Tu es un assistant pédagogique en cardiologie…",
  "input": "Quel est le rôle de l’auscultation cardiaque ?",
  "output": "L’auscultation permet d’écouter les bruits du cœur…",
  "meta": {
    "card_id": "ausc_01",
    "topic": "role",
    "source_ids": ["auscultation_cardiaque_normale"],
    "license": "…",
    "lang": "fr",
    "format": "qa_fr",
    "split": "train",
    "evidence": "extrait justificatif de la source"
  }
}
```

- `instruction` = rôle/consigne (envoyé comme message `system`) ;
- `input` = question ou contexte (message `user`) ;
- `output` = réponse attendue (message `assistant`) ;
- `meta` = traçabilité (source, licence, extrait, langue, split).

### 3.3 Construction

- 90 fiches de faits rédigées à partir des deux documents, avec **extrait justificatif** ;
- 6 tâches par fiche : Q/R FR, Q/R EN, explication patient FR/EN, question inverse FR/EN ;
- 50 guardrails FR/EN (refus de diagnostic, urgences, prescription, interprétation d'examens) ;
- découpage **stratifié par source et par fiche** : toutes les variantes d'une fiche restent dans
  le même split, ce qui évite les fuites ;
- volumes exacts : **450 / 70 / 70**.

Régénérer :

```bash
python llm_finetuning/data/build_dataset.py
python llm_finetuning/data/build_dataset.py --check
```

Le texte brut peut être ré-extrait avec :

```bash
pdftotext -layout rag_engine/Auscultation-cardiaque-normale-4-4.pdf \
  llm_finetuning/data/raw/auscultation_cardiaque_normale.txt
pdftotext -layout rag_engine/S127979602300089X.pdf \
  llm_finetuning/data/raw/esc_evaluation_cardiovasculaire.txt
```

### 3.4 Limites du dataset

- Deux documents seulement : couverture volontairement restreinte.
- Certaines réponses sont des **reformulations pédagogiques**, pas des recommandations
  individuelles.
- Le cours d'auscultation a des droits non explicités.
- Aucune validation clinique, aucune donnée patient, aucun diagnostic.

Voir [`data/dataset/dataset_card.md`](data/dataset/dataset_card.md).

---

## 4. Installation

### 4.1 Google Colab (recommandé)

```bash
pip install -U pip
pip install -r llm_finetuning/requirements-llm.txt
```

Unsloth peut nécessiter sa commande d'installation dédiée selon la version de CUDA :

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

En cas d'incompatibilité, le code bascule automatiquement sur **Transformers + PEFT +
bitsandbytes** (voir `src/model.py`).

### 4.2 Local (validation seulement)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pyyaml pytest rouge-score sacrebleu
python llm_finetuning/data/build_dataset.py --check
pytest llm_finetuning/tests -q
```

> Ne pas installer toute la stack d'entraînement sur la machine locale : le disque est limité.

---

## 5. Entraînement QLoRA

### 5.1 Configuration principale

`configs/qlora_qwen25_7b.yaml` :

| Paramètre | Valeur |
|---|---|
| Quantification | 4-bit NF4, double quant, compute `float16` (T4) |
| LoRA | r=16, α=32, dropout 0.05, `q/k/v/o/gate/up/down` |
| Séquence max | 2048 |
| Batch | 1 par device, accumulation 8 |
| LR | 2e-4, cosine, warmup 3 % |
| Époques | 2 (early stopping sur `eval_loss`) |
| Optimiseur | `paged_adamw_8bit` |
| Seed | 42 |

### 5.2 Vérification à sec (sans GPU)

```bash
python -m llm_finetuning.src.train \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml --dry-run
```

### 5.3 Entraînement

```bash
python -m llm_finetuning.src.train \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml
```

Smoke test rapide :

```bash
python -m llm_finetuning.src.train \
  --config llm_finetuning/configs/qlora_qwen25_1_5b.yaml --max-steps 20
```

Sorties :

```text
llm_finetuning/outputs/qwen25-7b-cardio/
├── adapter/                 # adaptateur LoRA + tokenizer
├── trainer_state.json
└── training_config.yaml
```

---

## 6. Suivi MLflow

- Expérience : **`cardioagent_llm`**.
- URI : `MLFLOW_TRACKING_URI` ou `mlflow.tracking_uri` de la config.
- Si le serveur est injoignable : repli automatique sur
  `sqlite:///llm_finetuning/mlruns.db`.

Métriques enregistrées :

- `loss`, `eval_loss`, `learning_rate`, `grad_norm`, `epoch` à chaque pas ;
- métriques finales d'entraînement et de validation ;
- empreintes SHA-256 des fichiers de dataset ;
- paramètres, versions des libs, GPU/VRAM, artefacts (`adapter/`, config, `trainer_state.json`).

Lancer un serveur local :

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --port 5000
```

Sur Colab, le repli SQLite permet de travailler hors ligne ; télécharger ensuite
`llm_finetuning/mlruns.db` et le dossier `mlruns/` pour consulter les runs localement.

---

## 7. Évaluation indépendante

```bash
# modèle fine-tuné
python -m llm_finetuning.src.evaluate \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml \
  --adapter llm_finetuning/outputs/qwen25-7b-cardio/adapter

# comparaison base vs fine-tuné
python -m llm_finetuning.src.evaluate \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml \
  --adapter llm_finetuning/outputs/qwen25-7b-cardio/adapter \
  --compare-base
```

Métriques : exact match normalisé, F1 tokens, ROUGE-L, BLEU, taux de redirection des guardrails.
Sorties : `metrics.json`, `predictions.jsonl`, `evaluation_report.md`.

> Rappel : ces métriques mesurent la **qualité de reformulation**, pas une performance clinique.

---

## 8. Export GGUF et intégration Ollama

### 8.1 Export (Colab/Linux)

Méthode `llama.cpp` (recommandée) :

```bash
git clone https://github.com/ggerganov/llama.cpp llm_finetuning/export/llama.cpp
cd llm_finetuning/export/llama.cpp
cmake -B build && cmake --build build --config Release -j
cd -

python -m llm_finetuning.src.export_gguf \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml \
  --adapter llm_finetuning/outputs/qwen25-7b-cardio/adapter \
  --output-dir llm_finetuning/export \
  --quantization q4_k_m
```

Méthode Unsloth :

```bash
python -m llm_finetuning.src.export_gguf \
  --config llm_finetuning/configs/qlora_qwen25_7b.yaml \
  --adapter llm_finetuning/outputs/qwen25-7b-cardio/adapter \
  --output-dir llm_finetuning/export \
  --method unsloth --quantization q4_k_m
```

Sorties :

```text
llm_finetuning/export/
├── cardio-llm-q4_k_m.gguf   # non versionné (*.gguf)
├── Modelfile                # généré depuis le template
└── MANIFEST.json            # taille, SHA-256, quantification
```

### 8.2 Ollama

```bash
ollama create cardio-llm -f llm_finetuning/export/Modelfile
ollama run cardio-llm
```

Test rapide (PowerShell) :

```powershell
.\llm_finetuning\scripts\ollama_smoke_test.ps1
```

Le `Modelfile` applique :

- un template ChatML ;
- `temperature 0.25`, `top_p 0.9`, `num_ctx 4096` ;
- un **SYSTEM prompt** avec les garde-fous (pas de diagnostic, pas de prescription,
  redirection en urgence, langues FR/EN).

> **Docker** : l'intégration au service `llm-server` sera faite **après validation du
> fine-tuning**. Le `docker-compose.yml` monte déjà `./llm_finetuning/export:/models:ro` et
> l'entrypoint cherche `Modelfile` + `cardio-llm.gguf` ; ces fichiers ne sont pas encore fournis.

### 8.3 Ressources locales

Sur la RTX 2050 (4 Go), un 7B `q4_k_m` ne tient pas entièrement en VRAM : Ollama fera du
déchargement CPU (lent mais fonctionnel). Pour une démo locale fluide, envisager
Qwen2.5-1.5B/3B ou un serveur GPU.

---

## 9. Tests

```bash
pytest llm_finetuning/tests -q
pytest llm_finetuning/tests --cov=llm_finetuning --cov-report=term-missing
```

Les tests couvrent :

- schéma et volumes du dataset ;
- absence de fuite entre splits ;
- ancrage des exemples aux sources ;
- validation des configurations ;
- format des prompts et présence des guardrails ;
- manifeste d'export et `.gitignore` ;
- déterminisme du générateur.

Aucun test ne nécessite de GPU ni de téléchargement de modèle.

---

## 10. Dépannage

| Symptôme | Piste |
|---|---|
| `CUDA out of memory` | réduire `max_seq_length` à 1024, batch 1, accumulation 16 ; utiliser `qwen25_1_5b` |
| Unsloth incompatible | le fallback PEFT s'active ; sinon épingler `torch`/`transformers` |
| `bitsandbytes` introuvable (Windows) | utiliser Colab/WSL2 |
| MLflow injoignable | repli SQLite automatique ; vérifier `MLFLOW_TRACKING_URI` |
| `llama-quantize` absent | compiler `llama.cpp` (section 8.1) |
| GGUF trop gros (> 100 Mo) | ne pas committer : `*.gguf` est déjà ignoré |
| Ollama lent | modèle partiellement sur CPU (4 Go VRAM) |
| `pdftotext` absent | installer Poppler, ou sauter la ré-extraction |

---

## 11. Éthique, licences et limites

- **Pas un dispositif médical** : aucune certification, aucune validation clinique.
- **Pas de diagnostic** : le modèle refuse et redirige les demandes de diagnostic.
- **Pas de prescription** : aucune posologie ni recommandation individuelle.
- **Urgences** : redirection systématique vers les secours.
- **Données** : aucun patient, aucune donnée personnelle ; ne pas soumettre de données
  identifiantes au modèle.
- **Licences** :
  - Qwen2.5-7B-Instruct : vérifier la licence sur la model card (Apache-2.0 pour cette taille
    au moment de la rédaction) ;
  - article ESC : CC BY-NC 4.0, usage non commercial avec attribution ;
  - cours d'auscultation : droits non explicités, usage académique interne.
- **Reproductibilité** : versions des libs, paramètres et empreintes de dataset sont journalisés
  dans MLflow.

---

## 12. Suite du projet

1. Valider le fine-tuning (pertes, évaluation, revue d'exemples).
2. Exporter le GGUF et créer le modèle Ollama `cardio-llm`.
3. **Ensuite seulement** : brancher `llm-server` (Docker), le RAG (`vector-db`) et
   l'orchestrateur multi-agents.
