# CardioAgent-Ops

**Analyse assistée de signaux phonocardiographiques (PCG) — SVM, arbre de décision et CNN, avec consensus multi-modèles et interface web.**

CardioAgent-Ops est un projet académique (TP3 — Data Mining) qui prend un enregistrement
cardiaque au format **WAV**, le prétraite de façon identique à l'entraînement, applique trois
algorithmes, agrège leurs prédictions au niveau de l'enregistrement et affiche un **consensus**.
Le tout est exposé par une API FastAPI et une interface web, avec suivi des expériences dans
**MLflow**.

> ⚠️ **Avertissement** : ce projet est un travail pédagogique. Il ne constitue ni un dispositif
> médical ni une aide au diagnostic. Ne jamais l'utiliser pour prendre une décision clinique.

---

## Sommaire

1. [Fonctionnalités](#1-fonctionnalités)
2. [Architecture](#2-architecture)
3. [Contenu du dépôt](#3-contenu-du-dépôt)
4. [Prérequis](#4-prérequis)
5. [Installation locale](#5-installation-locale)
6. [Données](#6-données)
7. [Entraîner les modèles](#7-entraîner-les-modèles)
8. [Lancer le projet en local](#8-lancer-le-projet-en-local)
9. [Lancer avec Docker](#9-lancer-avec-docker)
10. [Référence de l'API](#10-référence-de-lapi)
11. [Utilisation en ligne de commande](#11-utilisation-en-ligne-de-commande)
12. [Interface web](#12-interface-web)
13. [Configuration](#13-configuration)
14. [Modèles et performances](#14-modèles-et-performances)
15. [Dépannage](#15-dépannage)
16. [Fichiers volontairement non versionnés](#16-fichiers-volontairement-non-versionnés)
17. [Licence et avertissement](#17-licence-et-avertissement)

---

## 1. Fonctionnalités

- **Prétraitement unique** (`audio_engine/preprocess.py`) partagé par l'entraînement et
  l'inférence : rééchantillonnage à **2 000 Hz**, filtre passe-bande **25–400 Hz**, segmentation
  en fenêtres de **5 s** (50 % de chevauchement, 20 segments maximum), normalisation.
- **Trois modèles complémentaires** :
  - **SVM RBF** sur 46 caractéristiques par segment (20 MFCC moyenne + 20 MFCC écart-type,
    ZCR, centroïde spectral, RMS) ;
  - **arbre de décision** sur les mêmes caractéristiques ;
  - **CNN 2D** sur spectrogrammes **Mel 128 × 256**.
- **Agrégation par enregistrement** : moyenne des probabilités par segment, seuil propre à chaque
  modèle, puis **vote majoritaire** pour le consensus (égalité tranchée par la probabilité moyenne).
- **API REST** : `audio-engine` (inférence) et `api-gateway` (proxy, CORS, orchestration).
- **Interface web** autonome (`web/index.html`) : dépôt d'un WAV, découverte automatique du
  service disponible, affichage des trois résultats et du consensus.
- **MLflow** : chaque entraînement est tracé dans l'expérience `tp3_audio_engine` (paramètres,
  métriques, artefacts, modèle). Le service d'inférence fonctionne **sans** MLflow grâce aux
  artefacts locaux de `audio_engine/models/`.
- **Docker Compose** : pile complète (MLflow, Qdrant, Ollama, audio-engine, orchestrateur, gateway).

---

## 2. Architecture

```text
                       ┌──────────────────────────┐
                       │  Interface web            │
                       │  web/index.html (:5501)   │
                       └────────────┬─────────────┘
                                    │ HTTP (CORS *)
                                    ▼
                       ┌──────────────────────────┐
                       │  api-gateway  :8085       │
                       │  FastAPI — proxy / CORS   │
                       └───┬──────────┬───────────┘
                           │          │
             ┌─────────────┘          └────────────────┐
             ▼                                         ▼
  ┌───────────────────────┐                  ┌─────────────────────┐
  │  audio-engine :8001   │                  │ agent-orchestrator  │
  │  FastAPI + registry   │                  │ :8002 (stub)        │
  └───────────┬───────────┘                  └─────────────────────┘
              │
              │ charge les artefacts locaux
              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ best_svm.joblib  ·  best_tree.joblib  ·  best_cnn.pt         │
  │ cardio_model.onnx (endpoint historique /predict)             │
  └──────────────────────────────────────────────────────────────┘

  MLflow :5000  ← traçage des entraînements (optionnel à l'inférence)
  Qdrant :6333  ← base vectorielle (RAG, prévue)
  Ollama :11434 ← LLM local (prévu)
```

En local, l'interface peut appeler **directement** `audio-engine` sur `localhost:8001`.
En Docker, elle passe par la gateway sur `localhost:8085` (l'audio-engine reste interne au
réseau Docker).

---

## 3. Contenu du dépôt

```text
.
├── agents/main.py                  # orchestrateur d'agents (stub FastAPI, port 8002)
├── api-gateway/                    # passerelle FastAPI (proxy audio-engine/orchestrateur/MLflow)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── audio_engine/                   # cœur du projet : prétraitement + modèles + service
│   ├── preprocess.py               # pipeline audio partagé entraînement / inférence
│   ├── registry.py                 # chargement des 3 modèles, prédiction, consensus
│   ├── server.py                   # API FastAPI de l'audio-engine
│   ├── analyze_signal.py           # CLI d'analyse d'un WAV
│   ├── mlflow_utils.py             # helpers MLflow (scikit-learn et PyTorch)
│   ├── SVM.ipynb                   # entraînement du SVM
│   ├── Decisiontree.ipynb          # entraînement de l'arbre de décision
│   ├── CNN.ipynb                   # entraînement du CNN (Mel)
│   ├── configs/cnn_config.yaml
│   ├── src/                        # architecture et entraînement du CNN
│   └── models/                     # artefacts servis par l'API (+ métriques et figures)
├── dataset/dataset/                # jeux de données bruts (PhysioNet Challenge 2016, PASCAL)
├── docker/                         # Dockerfiles des services
├── docker-compose.yml              # pile complète
├── llm_finetuning/                 # emplacement des exports LLM (poids .gguf non versionnés)
├── ourdataset/                     # dataset du projet (normal/abnormal, plis, features)
├── rag_engine/                     # base du moteur RAG (à compléter)
├── tests/                          # emplacement des tests
├── web/index.html                  # interface web (fichier unique)
├── 01_eda_data_wrangling.ipynb     # analyse exploratoire et construction du dataset
├── import_signals.py               # import des WAV dans PostgreSQL
├── requirements-train.txt          # dépendances d'entraînement
├── README.md                       # ce document
└── README_SVM.md                   # mémo du pipeline SVM historique
```

---

## 4. Prérequis

| Outil | Version conseillée | Usage |
|---|---|---|
| Python | **3.10+** (3.10 utilisée par les images Docker) | entraînement, API, notebooks |
| pip | récent | installation des dépendances |
| Git | 2.x | clonage |
| Docker Desktop | optionnel | pile complète (`docker compose`) |
| GPU NVIDIA | optionnel | accélération Ollama (non requis pour l'inférence) |

> **Windows** : les commandes de ce README sont en PowerShell. Sous Linux/macOS, remplacer
> `.\.venv\Scripts\Activate.ps1` par `source .venv/bin/activate` et `curl.exe` par `curl`.

---

## 5. Installation locale

### 5.1 Cloner le dépôt

```powershell
git clone https://github.com/Ahmed-jarrayaa/cadrioagent-ops.git
cd cadrioagent-ops
```

> Le dépôt contient le dataset et les modèles : le clonage peut être volumineux.

### 5.2 Créer l'environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l'activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 5.3 Installer les dépendances

Dépendances d'entraînement (numpy, scipy, scikit-learn, librosa, MLflow, ONNX…) :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

Dépendances des services et du CNN :

```powershell
python -m pip install fastapi "uvicorn[standard]" python-multipart httpx
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Sur une machine sans GPU, l'index CPU de PyTorch évite de télécharger les paquets CUDA.
> Pour un GPU NVIDIA, installer la version CUDA correspondante depuis le site de PyTorch.

### 5.4 Vérifier l'installation

```powershell
python -c "import numpy, scipy, sklearn, librosa, mlflow, torch, fastapi; print('OK')"
```

---

## 6. Données

### 6.1 Dataset du projet (`ourdataset/`)

```text
ourdataset/
├── train/
│   ├── normal/            # 2 926 enregistrements WAV
│   ├── abnormal/          #   858 enregistrements WAV
│   ├── train_kfold1..3/   # plis d'entraînement (générés)
│   ├── test_kfold1..3/    # plis de test (générés)
│   ├── mel_features_svm_a.data
│   ├── mel_features_svm_b.data
│   └── report.txt
└── processed/
    ├── dataset_2000hz.npz     # 22 122 segments × 46 caractéristiques
    ├── metadata_clean.csv
    ├── class_weights.json
    └── mel_cache_cnn/         # cache Mel du CNN (non versionné, voir §16)
```

- **Entrée du système** : un fichier WAV mono (ou stéréo, converti en mono), durée minimale 1 s.
- **Cible** : classification binaire `normal` / `abnormal`.
- Le split d'entraînement suit une logique **par enregistrement** (pas par segment) afin d'éviter
  les fuites de données entre plis.

### 6.2 Jeux de données bruts (`dataset/`)

```text
dataset/dataset/
├── physionet/training-a ... training-f/   # PhysioNet / CinC Challenge 2016
└── pascal/                                # PASCAL Heart Sounds
```

Sources publiques :

- PhysioNet Challenge 2016 — *Classification of Normal/Abnormal Heart Sound Recordings* :
  <https://physionet.org/content/challenge-2016/1.0.0/>
- PASCAL Heart Sounds Challenge :
  <https://www.peterjbentley.com/heartchallenge/>

Respecter les licences et conditions de citation de chaque jeu de données.

### 6.3 Import dans PostgreSQL (optionnel)

`import_signals.py` insère les métadonnées des WAV dans une base PostgreSQL
(`localhost:5434`, base `heart_signals`). Schéma attendu :

```sql
CREATE TABLE signals (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    class           TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    duration        DOUBLE PRECISION,
    sampling_rate   INTEGER
);
```

```powershell
python import_signals.py
```

Adapter `DB_CONFIG` dans le script si nécessaire.

---

## 7. Entraîner les modèles

Les trois modèles sont entraînés dans les notebooks d'`audio_engine/` et enregistrés dans **la même
expérience MLflow** (`tp3_audio_engine`).

### 7.1 Démarrer MLflow

Option locale (SQLite + artefacts dans `./mlruns`) :

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Ou via Docker :

```powershell
docker compose up -d mlflow-server
```

Interface MLflow : <http://localhost:5000>

> Si le serveur MLflow est injoignable, `mlflow_utils.py` bascule automatiquement sur le stockage
> SQLite local `audio_engine/mlruns.db`. L'entraînement n'est donc pas bloqué hors ligne.

### 7.2 Exécuter les notebooks

| Notebook | Modèle | Run MLflow | Artefacts produits |
|---|---|---|---|
| `audio_engine/SVM.ipynb` | SVM RBF, 46 features/segment, `GridSearchCV` + `GroupKFold(3)` | `svm_rbf_segments` | `models/best_svm.joblib` + `best_svm_meta.json` |
| `audio_engine/Decisiontree.ipynb` | arbre de décision, mêmes features/plis | `decision_tree_gridsearch` | `models/best_tree.joblib` + `best_tree_meta.json` |
| `audio_engine/CNN.ipynb` | CNN 2D sur Mel 128 × 256 | `cnn_mel_segments` | `models/best_cnn.pt` + `best_cnn_meta.json` |

Ouvrir les notebooks dans Jupyter ou VS Code, sélectionner l'interpréteur du `.venv`, puis
exécuter les cellules dans l'ordre.

> Les artefacts `best_svm.joblib`, `best_tree.joblib` et `best_cnn.pt` sont **déjà présents** dans
> le dépôt : l'API peut démarrer sans réentraînement.

### 7.3 Résultats de référence

Moyenne des 3 plis pour le SVM et l'arbre, split test pour le CNN :

| Modèle | Macro précision | Exactitude | BA | Macro F1 | AUC |
|---|---|---|---|---|---|
| SVM RBF | 0,81 | 0,86 | 0,88 | 0,83 | 0,94 |
| Arbre de décision | 0,76 | 0,81 | 0,83 | 0,78 | 0,88 |
| CNN (Mel 128 × 256) | 0,80 | 0,83 | 0,88 | 0,81 | 0,93 |

Détails du CNN (split test, 568 enregistrements / 3 430 segments de 5 s, seuil réglé sur la
validation avec spécificité ≥ 0,80) : rappel 0,973, spécificité 0,788, **BA au niveau
enregistrement 0,820**, AUC 0,871. Environ 30 k paramètres, 15 epochs, torch CPU.

Les figures (matrices de confusion, courbes ROC/précision-rappel, interprétation de l'arbre) sont
dans `audio_engine/models/plots/`.

---

## 8. Lancer le projet en local

Trois processus suffisent (MLflow est optionnel). Utiliser **trois terminaux** avec le `.venv`
activé, depuis la racine du projet.

### Terminal 1 — MLflow (optionnel)

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

### Terminal 2 — audio-engine (obligatoire)

```powershell
cd audio_engine
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m uvicorn server:app --host 127.0.0.1 --port 8001
```

- Documentation interactive : <http://127.0.0.1:8001/docs>
- État des modèles : <http://127.0.0.1:8001/health>

> Lancer uvicorn **depuis** `audio_engine/` : `server.py` importe `registry` et `preprocess`
> comme modules du répertoire courant.
>
> `registry.py` recharge automatiquement un artefact dont le fichier a changé. En revanche, une
> modification du **code** (`server.py`, `registry.py`) nécessite de relancer uvicorn.

### Terminal 3 — interface web

```powershell
python -m http.server 5501 --bind 127.0.0.1 --directory .
```

Ouvrir : **<http://127.0.0.1:5501/web/index.html>**

### Terminal 3 bis — api-gateway (optionnel en local)

L'interface peut appeler directement l'audio-engine. Pour passer par la gateway :

```powershell
$env:AUDIO_ENGINE_URL = "http://localhost:8001"
$env:ORCHESTRATOR_URL = "http://localhost:8002"
$env:MLFLOW_URL = "http://localhost:5000"
python -m uvicorn main:app --host 127.0.0.1 --port 8085 --app-dir api-gateway
```

- Gateway : <http://127.0.0.1:8085/docs>

> Ne pas faire tourner en même temps Docker et les uvicorn sur les mêmes ports.

---

## 9. Lancer avec Docker

### 9.1 Démarrage complet

```powershell
docker compose up -d --build
docker compose ps
```

| Service | Port hôte | Rôle |
|---|---|---|
| `api-gateway` | **8085** | point d'entrée HTTP public (proxy) |
| `mlflow-server` | **5000** | suivi des expériences |
| `vector-db` (Qdrant) | **6333** | base vectorielle (RAG) |
| `llm-server` (Ollama) | **11434** | LLM local |
| `audio-engine` | interne `8001` | inférence SVM / arbre / CNN / ONNX |
| `agent-orchestrator` | interne `8002` | orchestrateur d'agents (stub) |

- API gateway : <http://localhost:8085/docs>
- MLflow : <http://localhost:5000>
- Interface web : servir `web/` puis laisser la découverte automatique choisir
  `http://localhost:8085`.

### 9.2 Commandes utiles

```powershell
docker compose logs -f audio-engine      # journaux de l'inférence
docker compose logs -f api-gateway
docker compose restart audio-engine
docker compose down                      # arrêt
docker compose down -v                   # arrêt + suppression des volumes
```

### 9.3 Remarques Docker

- **Sans GPU NVIDIA** : supprimer le bloc `deploy.resources` du service `llm-server` dans
  `docker-compose.yml`, sinon `docker compose up` peut échouer.
- Le service `llm-server` attend un `Modelfile` et un poids `.gguf` dans
  `llm_finetuning/export/` (non versionnés). Sans eux, Ollama démarre mais le modèle
  `cardio-llm` n'est pas créé ; cela ne bloque pas l'inférence audio.
- `agent-orchestrator` est un **stub** : `POST /analyze` renvoie simplement la charge reçue.
- Les modèles sont montés en lecture seule depuis `./audio_engine/models`.
- L'audio-engine n'expose pas de port hôte : passer par la gateway (`8085`). Pour l'exposer
  directement, ajouter `ports: ["8001:8001"]` au service.

---

## 10. Référence de l'API

### 10.1 audio-engine (`:8001`)

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | liste des endpoints |
| `GET` | `/health` | état du modèle ONNX et des trois modèles TP3 |
| `GET` | `/models` | seuils, runs MLflow, métriques disponibles |
| `POST` | `/predict` | modèle ONNX historique (`models/cardio_model.onnx`) |
| `POST` | `/predict_all?families=svm,tree,cnn` | les trois modèles + consensus |
| `POST` | `/predict_one?family=svm` | un seul modèle (`svm`, `tree` ou `cnn`) |

Exemples :

```powershell
curl.exe -s http://127.0.0.1:8001/health
curl.exe -s http://127.0.0.1:8001/models

curl.exe -s -X POST "http://127.0.0.1:8001/predict_all?families=svm,tree,cnn" `
  -F "file=@ourdataset/train/abnormal/112_1306243000964_A.wav"

curl.exe -s -X POST "http://127.0.0.1:8001/predict_one?family=svm" `
  -F "file=@ourdataset/train/normal/101_1305030823364_B.wav"
```

Réponse simplifiée de `/predict_all` :

```json
{
  "filename": "112_1306243000964_A.wav",
  "duration_s": 12.5,
  "sample_rate_used": 2000,
  "n_segments": 4,
  "consensus": {
    "prediction": "abnormal",
    "votes": { "normal": 1, "abnormal": 2 },
    "models_ok": 3,
    "models_total": 3,
    "mean_prob_abnormal": 0.74,
    "mean_threshold": 0.30
  },
  "analysis": [
    {
      "key": "svm",
      "family": "SVM",
      "label": "SVM (rbf)",
      "prediction": "abnormal",
      "prob_abnormal_mean": 0.87,
      "prob_abnormal_max": 0.95,
      "threshold": 0.099,
      "n_segments": 4,
      "status": "ok",
      "mlflow_run_name": "svm_rbf_segments"
    }
  ]
}
```

Erreurs fréquentes :

| Code | Cause |
|---|---|
| `400` | extension autre que `.wav`, famille inconnue |
| `413` | fichier plus gros que `MAX_UPLOAD_MB` (25 Mo par défaut) |
| `422` | audio illisible, trop court (< 1 s) ou silencieux |
| `500` | artefact de modèle absent ou erreur d'inférence |
| `503` | modèle ONNX non chargé (`/predict` uniquement) |

### 10.2 api-gateway (`:8085`)

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | informations et liste des endpoints |
| `GET` | `/health` | état de la gateway, de l'audio-engine, de l'orchestrateur et de MLflow |
| `POST` | `/predict` | proxy direct vers l'audio-engine `/predict` (ONNX) |
| `POST` | `/analyze_all?families=svm,tree,cnn` | proxy vers `/predict_all` |
| `POST` | `/predict_one?family=svm` | proxy vers `/predict_one` |
| `POST` | `/analyze_pcg` | ONNX puis orchestrateur (`/analyze`) |
| `GET` | `/models` | proxy vers `/models` |
| `GET` | `/mlflow/experiments` | liste des expériences MLflow |

```powershell
curl.exe -s http://127.0.0.1:8085/health

curl.exe -s -X POST "http://127.0.0.1:8085/analyze_all?families=svm,cnn" `
  -F "file=@ourdataset/train/abnormal/112_1306243000964_A.wav"

curl.exe -s -X POST "http://127.0.0.1:8085/predict_one?family=tree" `
  -F "file=@ourdataset/train/normal/101_1305030823364_B.wav"
```

---

## 11. Utilisation en ligne de commande

Sans serveur HTTP, la même logique est disponible dans `audio_engine/` :

```powershell
cd audio_engine

python analyze_signal.py ..\ourdataset\train\abnormal\112_1306243000964_A.wav
python analyze_signal.py ..\ourdataset\train\abnormal\112_1306243000964_A.wav --json
python analyze_signal.py ..\ourdataset\train\normal\101_1305030823364_B.wav --families svm,cnn
```

Sortie lisible :

```text
==========================================================================
ANALYSE PCG — SVM / arbre de décision / CNN (audio_engine)
==========================================================================
Fichier  : 112_1306243000964_A.wav
Durée    : 12.5 s | 4 segment(s) | sr=2000 Hz
MLflow   : http://localhost:5000 / tp3_audio_engine
--------------------------------------------------------------------------
[SVM (rbf)] run=svm_rbf_segments (…)
   prédiction=abnormal  P(abnormal)=0.870  seuil=0.099  max=0.950
...
--------------------------------------------------------------------------
CONSENSUS : abnormal  votes={'normal': 1, 'abnormal': 2}  modèles OK=3/3
```

`registry.py` expose la même interface :

```powershell
python registry.py ..\ourdataset\train\abnormal\112_1306243000964_A.wav --json
```

---

## 12. Interface web

L'interface est un fichier HTML autonome : `web/index.html`.

1. Démarrer l'audio-engine (ou la gateway).
2. Servir le dossier racine :
   ```powershell
   python -m http.server 5501 --bind 127.0.0.1 --directory .
   ```
3. Ouvrir <http://127.0.0.1:5501/web/index.html>.

Fonctionnement :

- **découverte automatique** du service disponible, dans cet ordre :
  `localhost:8085` (gateway), `127.0.0.1:8085`, `localhost:8001` (audio-engine),
  `127.0.0.1:8001`, puis l'origine courante ;
- dépôt ou sélection d'un **WAV** ;
- appel de `/analyze_all` (gateway) ou `/predict_all` (audio-engine) ;
- affichage des trois cartes de résultats, du **consensus**, des métriques de référence et des
  détails techniques (probabilités par segment, caractéristiques moyennes).

> L'API autorise toutes les origines (`CORS *`) : l'interface peut être ouverte depuis un autre
> serveur local ou via `file://`.

---

## 13. Configuration

Variables d'environnement reconnues :

| Variable | Défaut | Service | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` (`http://mlflow-server:5000` en Docker) | audio-engine, registry | serveur MLflow à interroger |
| `TP3_EXPERIMENT` | `tp3_audio_engine` | registry | expérience contenant les runs |
| `MODEL_DIR` | `audio_engine/models` | registry | dossier des artefacts TP3 |
| `MODEL_PATH` | `models/cardio_model.onnx` | audio-engine | modèle ONNX historique |
| `MAX_UPLOAD_MB` | `25` | audio-engine | taille maximale d'un WAV |
| `MLFLOW_LOGGING` | `1` | audio-engine | `0` pour désactiver la trace des inférences ONNX |
| `AUDIO_ENGINE_URL` | `http://audio-engine:8001` | api-gateway | URL de l'audio-engine |
| `ORCHESTRATOR_URL` | `http://agent-orchestrator:8002` | api-gateway | URL de l'orchestrateur |
| `MLFLOW_URL` | `http://mlflow-server:5000` | api-gateway | URL de MLflow |
| `LLM_URL` | `http://llm-server:11434` | agent-orchestrator | URL d'Ollama |
| `LLM_MODEL` | `cardio-llm` | agent-orchestrator | nom du modèle Ollama |
| `QDRANT_URL` | `http://vector-db:6333` | agent-orchestrator | URL de Qdrant |

Exemple PowerShell :

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MAX_UPLOAD_MB = "50"
```

> Utiliser `localhost` plutôt que `127.0.0.1` dans `MLFLOW_TRACKING_URI` : selon la configuration,
> le serveur MLflow n'écoute qu'en IPv6. `registry.py` teste automatiquement les deux variantes.

---

## 14. Modèles et performances

| Clé API | Famille | Fichier | Seuil | Agrégation |
|---|---|---|---|---|
| `svm` | SVM RBF | `models/best_svm.joblib` | `best_svm_meta.json` (≈ 0,099) | sigmoïde du `decision_function`, moyenne |
| `tree` | arbre de décision | `models/best_tree.joblib` | `best_tree_meta.json` (0,5 par défaut) | `predict_proba`, moyenne |
| `cnn` | CNN Mel 128 × 256 | `models/best_cnn.pt` | `best_cnn_meta.json` (≈ 0,309) | softmax, moyenne |

**Consensus** : chaque modèle vote `normal` ou `abnormal` selon son propre seuil. Le consensus est
la majorité ; en cas d'égalité, la probabilité moyenne d'anomalie départage.

Le endpoint `/models` renvoie l'état réel (fichier présent, seuil, run MLflow, métriques) :

```powershell
curl.exe -s http://127.0.0.1:8001/models
```

Le modèle ONNX historique (`models/cardio_model.onnx`) reste servi par `POST /predict` pour
compatibilité ; il prend en entrée les 46 caractéristiques par segment et applique le seuil de
`cardio_model_meta.json`.

---

## 15. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `ModuleNotFoundError: registry` | lancer uvicorn depuis `audio_engine/` ou utiliser `--app-dir audio_engine` |
| `503 Modèle ONNX non chargé` | `models/cardio_model.onnx` ou `cardio_model_meta.json` absent ; les endpoints TP3 restent utilisables |
| `FileNotFoundError: best_cnn.pt absent` | relancer `CNN.ipynb` pour régénérer l'artefact |
| `[audio-engine] MLflow non chargé` | serveur MLflow arrêté : sans impact sur l'inférence (mode local) |
| `422 Fichier audio illisible` | WAV corrompu, moins de 1 s ou silencieux |
| `413 Fichier trop volumineux` | augmenter `MAX_UPLOAD_MB` |
| MLflow injoignable en local | utiliser `localhost` au lieu de `127.0.0.1` (IPv6), ou `docker compose up -d mlflow-server` |
| Plantage DLL `matplotlib` + `onnx` sous Windows | exécuter les étapes dans des processus séparés (voir `README_SVM.md`) |
| `torch` introuvable | `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `libsndfile` manquant (Linux) | `sudo apt-get install libsndfile1` (déjà fait dans l'image Docker) |
| `docker compose` échoue sur `llm-server` sans GPU | supprimer le bloc `deploy.resources` du service |
| Port déjà utilisé | `Get-NetTCPConnection -LocalPort 8001` puis arrêter le processus, ou changer de port |
| Interface bloquée sur « Connexion… » | vérifier `/health` sur `8001` et `8085`, et les CORS |

---

## 16. Fichiers volontairement non versionnés

| Chemin | Raison | Comment le régénérer |
|---|---|---|
| `audio_engine.zip` | archive locale redondante avec `audio_engine/` | — |
| `CardioAgent-Ops-interface-enrichie.html` | variante d'interface conservée en local | — |
| `audio_engine/models/tp3_cnn_mels.npy` | **1,38 Go** : dépasse la limite GitHub de 100 Mo | relancer `audio_engine/CNN.ipynb` |
| `ourdataset/processed/mel_cache_cnn/` | caches Mel jusqu'à 955 Mo | relancer `audio_engine/CNN.ipynb` |
| `mlruns/`, `mlruns.db`, `mlflow.db` | état local MLflow | relancer les entraînements |
| `.venv/`, `__pycache__/`, `*.pyc` | environnements et caches | recréer le venv |
| `.env`, `*.gguf` | secrets et poids de LLM volumineux | fournir localement |

Le `.gitignore` détaille ces exclusions. Le reste du projet (code, notebooks, dataset, modèles
légers, figures) est versionné.

---

## 17. Licence et avertissement

Projet réalisé dans un cadre **académique** (Semestre 1 — 3GMAM, module Data Mining).

Les jeux de données appartiennent à leurs auteurs respectifs (PhysioNet, PASCAL) et sont soumis à
leurs propres licences. Les modèles et le code sont fournis **à des fins de démonstration**, sans
aucune garantie.

**Ce logiciel n'est pas un dispositif médical et ne doit pas être utilisé pour poser un diagnostic
ou orienter une prise en charge clinique.**
