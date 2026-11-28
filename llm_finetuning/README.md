# LLM Fine-Tuning — CardioAgent-Ops

Adaptation **QLoRA 4-bit** d'un modèle open source sur un corpus pédagogique de cardiologie,
avec suivi **MLflow**, évaluation indépendante et export **GGUF** pour **Ollama**.

**Modèle livré et réellement entraîné : Qwen2.5-1.5B-Instruct**, sur la machine locale
(RTX 2050, 4 Go). Le **7B** est la configuration cible, fournie et prête, mais non exécutée :
elle nécessite un GPU ≥ 12 Go (par exemple Colab T4). Cette décision est documentée en §1 et §5.4.

> ⚠️ **Avertissement** : ce module est un travail **académique**. Le modèle obtenu n'est **pas un
> dispositif médical**, ne pose pas de diagnostic et ne doit jamais servir à une décision
> clinique. Il ne remplace ni l'examen physique ni un professionnel de santé.

---

## 1. Périmètre et contraintes

| Élément | Décision |
|---|---|
| Modèle livré | **Qwen2.5-1.5B-Instruct** (QLoRA 4-bit, entraîné en local) |
| Configuration cible | **Qwen2.5-7B-Instruct** (prête, non exécutée : GPU ≥ 12 Go requis) |
| Exécution d'entraînement | Locale (1.5B, RTX 2050 4 Go) ; Colab T4/L4 prévu pour le 7B |
| Repli | Le 1.5B sert à la fois de repli et de modèle de démonstration livré |
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

Le 7B reste donc sur Colab **si un GPU adapté est disponible**. En l'absence de Colab, tout le
pipeline a été validé sur le 1.5B local : dataset, entraînement réel (112 pas, 38 min),
évaluation indépendante, export GGUF `q4_k_m`, test Ollama et service Docker `llm-server`.
Le 7B n'est pas nécessaire pour démontrer le pipeline, mais il demeure la cible de qualité.

---

## 2. Arborescence

```text
llm_finetuning/
├── README.md                     # ce document
├── requirements-llm.txt          # stack Colab/Linux
├── requirements-llm-local.txt    # stack Windows locale (sans Unsloth)
├── configs/
│   ├── qlora_qwen25_7b.yaml      # configuration principale
│   ├── qlora_qwen25_1_5b.yaml    # repli / smoke test
│   └── qlora_qwen25_1_5b_local.yaml  # GPU 4 Go (Windows natif)
├── data/
│   ├── sources.json              # registre des sources et licences
│   ├── seed_facts.jsonl          # 90 fiches de faits ancrées (FR/EN)
│   ├── seed_guardrails.jsonl     # 50 consignes de sécurité (FR/EN)
│   ├── build_dataset.py          # génération + découpage + contrôles
│   ├── verify_sources.py         # empreintes des PDF + texte extrait
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
│   ├── train_local.ps1
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

### 4.1 Google Colab (optionnel, pour le 7B)

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

### 4.2 Local (modèle livré 1.5B)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pyyaml pytest rouge-score sacrebleu
python llm_finetuning/data/build_dataset.py --check
pytest llm_finetuning/tests -q
```

> La stack locale ci-dessous suffit au 1.5B livré. Ne pas installer le 7B sur cette machine :
> le disque et la VRAM sont limités (voir §1 et §5.4).

### 4.3 Notebook portable

`notebooks/colab_finetune.ipynb` fonctionne **sur Colab et en local** :

- la première cellule détecte la racine du dépôt (ou clone dans `/content` sur Colab) et s'y place ;
- la configuration est choisie selon la VRAM : **7B si ≥ 12 Go**, **1.5B local sinon** ;
- les dépendances installées dépendent de l'OS : `requirements-llm.txt` (Linux/Colab) ou
  `requirements-llm-local.txt` (Windows) ;
- l'export GGUF fonctionne aussi sur Windows avec les binaires llama.cpp précompilés (§8.1),
  sans installer `cmake`.

Les sorties d'exécution ne sont pas versionnées, et les chemins sont relatifs au dépôt.

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

### 5.4 Entraînement local 1.5B (GPU 4 Go)

Si Colab n'est pas disponible, le 1.5B peut être entraîné **réellement** sur la machine locale
(RTX 2050, 4 Go). Le 7B, lui, reste impossible : 4 Go de VRAM et ~16 Go de disque ne suffisent pas.

```powershell
# 1) Remplacer le torch CPU par un build CUDA (~2,4 Go, une seule fois)
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu121

# 2) Stack PEFT/TRL/bitsandbytes (sans Unsloth, non supporté sur Windows natif)
.\.venv\Scripts\python.exe -m pip install --no-cache-dir `
  -r llm_finetuning\requirements-llm-local.txt

# 3) Vérification
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# 4) Entraînement
.\.venv\Scripts\python.exe -m llm_finetuning.src.train `
  --config llm_finetuning\configs\qlora_qwen25_1_5b_local.yaml
```

Différences avec la config 7B : `max_seq_length: 384`, LoRA **sur l'attention seule**
(`q/k/v/o_proj`, ~5,5 M paramètres entraînables au lieu de 18,5 M), `gradient_accumulation_steps: 8`,
`dataloader_num_workers: 0`, `save_total_limit: 1`, évaluation une fois par époque (`eval_steps: 50`).

| Ressource | Mesuré |
|---|---|
| VRAM | ~3,4 Go / 4 Go (pic) |
| Durée (2 époques, 112 pas) | **37 min 49 s** |
| `train_loss` | 1,40 (moyenne sur le run) |
| `eval_loss` | 2,20 → **1,148** |
| Disque | torch CUDA + modèle ≈ 7 Go |

Pourquoi ces réglages : au-delà de ~3,7 Go utilisés, le pilote Windows **paginait la VRAM sur le
disque** (pas de 7 s → 60-100 s, évals de 20 s → 120 s) sans jamais lever d'OOM. Le passage à
`max_seq_length: 384` et au LoRA attention seule a ramené le pic à ~3,4 Go.

En cas de `CUDA out of memory` : fermer les jeux/navigateur (ils consomment de la VRAM partagée),
descendre `max_seq_length` à 256, puis `gradient_accumulation_steps` à 16. Le code libère déjà le
cache CUDA après chaque évaluation et tous les 25 pas (`ReleaseCudaCallback`), et n'accumule pas les
logits en éval (`prediction_loss_only: true`).

> L'export GGUF local demande ~7 Go au pic (fusion fp16 + GGUF f16). Il a été réalisé avec
> succès avec ~9,6 Go libres : `--merge-device cpu` charge le base en fp32 pour la fusion, puis
> repasse en fp16, et les intermédiaires sont supprimés automatiquement après quantification.

> ⚠️ Cette installation modifie le `.venv` partagé : elle remplace `torch` par un build CUDA et
> change les versions de `transformers`/`numpy`, ce qui peut impacter la partie audio. Pour éviter
> tout effet de bord, créer un environnement dédié :
>
> ```powershell
> python -m venv llm_finetuning\.venv-llm
> .\llm_finetuning\.venv-llm\Scripts\Activate.ps1
> pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
> pip install --no-cache-dir -r llm_finetuning\requirements-llm-local.txt
> ```
>
> `llm_finetuning/.venv-llm/` est ignoré par Git.

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

Résultats conservés dans le dépôt (les poids et sorties brutes sont ignorés par Git) :

| Run | Rapport |
|---|---|
| 1.5B local, 2 époques (38 min) | `llm_finetuning/reports/local_1_5b_evaluation.md` + `metrics_local_1_5b.json` |

Le rapport local documente une progression de F1/ROUGE-L/BLEU **mais un guardrail en échec
sur 4** (cas `guard_fr_02`), avec n=4 : signal à ne pas surinterpréter. Le 7B n'ayant pas été
exécuté, la mitigation retenue est une **défense en profondeur** : prompt système renforcé du
`Modelfile` + vérification automatique des refus dans `scripts/ollama_smoke_test.ps1`.

---

## 8. Export GGUF et intégration Ollama

### 8.1 Export (local Windows, sans `cmake`)

Le modèle livré a été exporté sur Windows avec les binaires llama.cpp **précompilés** :
aucun Visual Studio ni `cmake` n'est nécessaire.

```powershell
# 1) sources llama.cpp (convert_hf_to_gguf.py) + binaire précompilé llama-quantize.exe
git clone --depth 1 --branch b12728 https://github.com/ggml-org/llama.cpp.git `
  llm_finetuning\export\llama.cpp
# Télécharger llama-b12728-bin-win-cpu-x64.zip depuis les releases GitHub et
# extraire son contenu dans llm_finetuning\export\llama.cpp

# 2) dépendance Python du convertisseur
.\.venv\Scripts\python.exe -m pip install --no-cache-dir "gguf>=0.10"

# 3) fusion CPU (indispensable sur 4 Go de VRAM) + conversion + quantification
.\.venv\Scripts\python.exe -m llm_finetuning.src.export_gguf `
  --config llm_finetuning\configs\qlora_qwen25_1_5b_local.yaml `
  --adapter llm_finetuning\outputs\qwen25-1.5b-local\adapter `
  --output-dir llm_finetuning\export `
  --quantization q4_k_m --merge-device cpu
```

Sur Colab/Linux, garder `--merge-device auto` et utiliser la config 7B. La méthode Unsloth
reste disponible avec `--method unsloth` si l'adaptateur a été entraîné par Unsloth.

Sorties :

```text
llm_finetuning/export/
├── cardio-llm-q4_k_m.gguf   # 940,4 Mo, non versionné (*.gguf)
├── Modelfile                # généré depuis le template, garde-fous renforcés
└── MANIFEST.json            # taille, SHA-256, quantification
```

SHA-256 du GGUF livré :
`e8d53970f45d67b3ccc612d8777af23854d0a25abba85e70e11711fc0bf0dfcd`.

### 8.2 Ollama et service Docker

```powershell
ollama create cardio-llm -f llm_finetuning\export\Modelfile
.\llm_finetuning\scripts\ollama_smoke_test.ps1
```

Le script de smoke test utilise l'API HTTP d'Ollama (plus fiable que `ollama run` sous Windows),
crée le modèle, envoie 5 prompts et **échoue si un refus de diagnostic ou une redirection en
urgence manque**. Rapport : `llm_finetuning/reports/ollama_smoke_1_5b.json`.

Le `Modelfile` applique :

- un template ChatML ;
- `temperature 0.25`, `top_p 0.9`, `num_ctx 4096` ;
- un **SYSTEM prompt** renforcé : pas de diagnostic, pas de prescription, refus explicite
  et redirection en urgence, langues FR/EN.

Résultat mesuré : **3/3 garde-fous respectés** en local (FR et EN) et dans le conteneur Docker.
Les réponses factuelles détaillées restent à valider via le RAG : le 1.5B peut encore produire
une affirmation inexacte (ex. description erronée de B1). Le LLM ne doit pas être la source
unique d'un fait médical.

**Docker** : le service `llm-server` est maintenant intégré et vérifié. Le compose monte
`./llm_finetuning/export:/models:ro` ; l'entrypoint crée `cardio-llm` depuis `/models/Modelfile`
au démarrage du conteneur.

```powershell
docker compose up -d --force-recreate llm-server
docker logs -f llm-server
```

Rapport du test via le service : `llm_finetuning/reports/docker_llm_server_smoke_1_5b.json`.

> ⚠️ Ollama local et `llm-server` utilisent tous deux le port `11434` : arrêter l'un avant de
> démarrer l'autre.

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
  - Qwen2.5-1.5B-Instruct / 7B-Instruct : vérifier la licence sur la model card (Apache-2.0
    pour ces tailles au moment de la rédaction) ;
  - article ESC : CC BY-NC 4.0, usage non commercial avec attribution ;
  - cours d'auscultation : droits non explicités, usage académique interne.
- **Reproductibilité** : versions des libs, paramètres et empreintes de dataset sont journalisés
  dans MLflow.

---

## 12. Suite du projet

Étapes réalisées pour le 1.5B livré :

1. ✅ Entraînement QLoRA (112 pas, `eval_loss` 1,148) et évaluation 70 exemples.
2. ✅ Export GGUF `q4_k_m` (SHA-256 vérifié) et modèle Ollama `cardio-llm`.
3. ✅ Smoke test des garde-fous en local et dans `llm-server` (Docker).
4. ✅ Documentation des limites et des mesures dans `llm_finetuning/reports/`.
5. ✅ RAG (`rag_engine/`, Qdrant) et orchestrateur 3 agents (`agents/`) branchés sur
   `cardio-llm` ; 8/8 cas validés (voir `agents/reports/agent_smoke_1_5b.md`).

Reste à faire si un GPU ≥ 12 Go devient disponible :

1. exécuter `qlora_qwen25_7b.yaml` sur Colab, puis reproduire §7 et §8 ;
2. comparer les métriques du 7B à celles du 1.5B (§7).
