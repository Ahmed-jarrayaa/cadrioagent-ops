# CardioAgent-Ops — Rapport technique consolidé

> Projet académique (Semestre 1 — 3GMAM, Data Mining). **Aucun élément de ce dépôt n'est un
> dispositif médical** : les scores audio sont des tests de dépistage et l'assistant ne pose
> jamais de diagnostic ni de prescription.

Ce rapport consolide les preuves produites aux jalons 1 à 8. Les chiffres cités proviennent
des rapports JSON/Markdown versionnés et de l'exécution réelle sur la machine de développement.

---

## 1. Périmètre et données

- Deux seules sources documentaires : `rag_engine/Auscultation-cardiaque-normale-4-4.pdf` et
  `rag_engine/S127979602300089X.pdf` (article CC BY-NC 4.0 ; cours à droits non explicités,
  usage académique interne). Aucune donnée patient réelle, aucun fait médical inventé.
- Audio : fusion PASCAL + PhysioNet, rééchantillonnage 2 000 Hz, filtre passe-bande 25–400 Hz,
  fenêtres de 5 s avec recouvrement 50 %, 46 caractéristiques (20 MFCC moyens + 20 écarts-types
  + ZCR, centroïde spectral, RMS). Le prétraitement de `audio_engine/preprocess.py` est celui
  du notebook `01_eda_data_wrangling.ipynb` ; il n'a pas été modifié.
- Jeu LLM : 450 entraînement / 70 validation / 70 test (split sans fuite par document).

## 2. Jalon 2 — Benchmark audio (5 algorithmes)

Métriques moyennes des plis (GroupKFold par enregistrement), lues dans les métadonnées MLflow
de `audio_engine/models/best_*_meta.json` :

| Modèle | Accuracy | Balanced accuracy | Macro-recall | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM RBF | **0,859** | 0,883 | 0,883 | **0,831** | **0,936** |
| CNN 2D (Mel) | 0,840 | **0,885** | **0,885** | 0,812 | — |
| Transfer Learning VGG16 | 0,840 | 0,885 | 0,885 | 0,811 | — |
| Transfer Learning ResNet50 | 0,836 | 0,877 | 0,877 | 0,806 | — |
| Transfer Learning ResNet18 | 0,835 | 0,876 | 0,876 | 0,805 | — |
| Transformer-CNN Hybride | 0,746 | 0,685 | 0,685 | 0,670 | — |
| Arbre de décision | 0,811 | 0,832 | 0,832 | 0,779 | 0,883 |

Le **SVM RBF** est le modèle exporté en ONNX (`cardio_model.onnx`, 44,7 Mo) : meilleur ROC-AUC
et meilleur F1, avec un seuil stable. Les cinq familles restent servies par `/predict_all` et
`/predict_one` ; chaque run est tracé dans l'expérience MLflow `tp3_audio_engine`. Le CNN et le
transfer learning atteignent un rappel comparable mais avec des artefacts 100 à 500 fois plus
lourds.

## 3. Jalon 3 — Fine-tuning QLoRA et registre

| Élément | Résultat |
|---|---|
| Base livrée | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA 4-bit) |
| Entraînement | 112 pas, `eval_loss` 1,148, run MLflow `dc59b2d7b9ac4e15811df74c238b8e5b` |
| Test indépendant | base vs fine-tuné dans `llm_finetuning/reports/local_1_5b_evaluation.md` |
| Export | `cardio-llm-q4_k_m.gguf`, 986 048 320 octets (940,4 Mo), SHA-256 `e8d53970f45d67b3ccc612d8777af23854d0a25abba85e70e11711fc0bf0dfcd` |
| Ollama | `cardio-llm:latest` servi par `llm-server` (vérifié en Docker) |

La configuration 7B (`configs/qlora_qwen25_7b.yaml`) reste la cible du PDF mais n'a pas pu être
exécutée localement : la machine dispose d'une **RTX 2050 de 4 Go de VRAM**. La fusion du 1.5B a
été faite sur CPU (`--merge-device cpu`) pour ne pas saturer cette VRAM. L'évaluation indépendante
utilise F1 / ROUGE-L / BLEU ; `exact_match = 0` car les réponses sont paraphrasées. Ces métriques
n'ont **aucune valeur clinique** : le LLM ne remplace ni le RAG ni un professionnel.

## 4. Jalons 4 et 5 — RAG et multi-agents

- **RAG** : `BAAI/bge-m3` (1 024 dimensions, similarité cosinus), collection Qdrant
  `cardio_docs`, **66 passages**, seuil `min_score = 0,35`, **6/6 questions validées**
  (`rag_engine/reports/rag_evaluation.json`). Les IDs de passages sont des SHA-256 stables :
  l'indexation est idempotente.
- **Graphe LangGraph** : `START → Analyste → Cardiologue RAG → Auditeur → END`, contrats
  Pydantic `AnalyzeRequest` / `AnalyzeResponse`.
- **Garde-fous** : diagnostic, prescription et urgence produisent un refus **déterministe** sans
  appel au LLM ; absence de source ⇒ abstention ; l'auditeur remplace toute réponse non ancrée
  ou contenant un dosage interdit ; un score audio est toujours requalifié en dépistage.
- Preuve réelle : `agents/reports/agent_smoke_1_5b.json`, **8/8 audits OK**. Limitations
  documentées : audit lexical (pas de vérification de la vérité médicale) et avertissements
  `cross_lingual_grounding` / `language_mismatch` quand le 1.5B répond en français à une question
  anglaise. L'interface web (`web/index.html`) expose désormais ce chat : elle appelle `/analyze`
  et affiche la réponse, les sources, l'audit et les garde-fous, ainsi qu'un panneau de liens vers
  les rapports et preuves du projet.
- **Rapport d'analyse personnalisé** : la gateway expose `POST /analyze_all_report`. Elle appelle
  `/predict_all` avec explicitement `families=svm,tree,cnn,tl,hybrid`, vérifie que les cinq familles
  sont présentes dans la réponse, puis transmet un résumé des résultats réels à l'orchestrateur avec
  `mode=report`. L'agent construit les sections factuelles (prédictions, scores, seuils,
  caractéristiques, votes, divergences), l'interprétation et la conclusion à partir de ces seules
  valeurs ; aucun chiffre n'est inventé. La conclusion met en regard le vote des modèles et un
  passage réellement récupéré dans les deux documents (citation `[1]`). Le LLM est limité à une
  ligne structurée de confirmation des faits ; une instruction hors périmètre est bloquée
  avant l'appel (`intent=out_of_scope`) ; le LLM ne reçoit que des instructions déjà filtrées. Si la
  confirmation contredit le workflow, elle est rejetée (`llm_confirmation_rejected`). En cas
  d'indisponibilité du LLM, l'audit signale `llm_unavailable` et le rapport reste disponible.
- **Traçabilité des 6 services** : `GET /workflow/services` expose l'état de `api-gateway`,
  `audio-engine`, `mlflow-server`, `agent-orchestrator`, `vector-db` et `llm-server` (les deux
  derniers via la santé de l'orchestrateur). La réponse de `/analyze_all_report` contient en plus
  `workflow_trace` (statut, détail et durée mesurée pour chaque service). L'interface a supprimé
  le panneau « Jalons 3 à 8 · Preuves » et affiche ce parcours à la place.
- **Validation du prompt avant prédiction** : la gateway appelle en premier
  `POST /check_prompt` de l'orchestrateur. Cette vérification déterministe contrôle que
  l'instruction porte sur l'analyse/la classification des signaux cardiaques et qu'elle ne
  demande ni diagnostic, ni prescription, ni gestion d'urgence. Elle n'effectue aucun appel
  RAG, LLM ou audio-engine. Si le prompt est rejeté, le workflow s'arrête immédiatement avec
  un message FR/EN invitant à reformuler ; `/predict_all` n'est jamais appelé. La route
  non diffusée `/analyze_all_report` applique la même règle. Côté interface, une erreur de
  validation annule tout repli vers `/predict_all` ; si les routes de rapport sont absentes,
  un appel explicite à `/check_prompt` est exigé avant toute prédiction brute. La validation
  est aussi imposée côté passerelle sur `/predict`, `/analyze_all`, `/predict_one` et
  `/analyze_pcg` : aucune de ces routes ne peut déclencher l'audio-engine avec un prompt
  hors domaine, vide ou indisponible.
- **Workflow temps réel (SSE)** : `POST /analyze_all_report/stream` diffuse les événements
  `step` (service, statut `running` / `ok` / `erreur`, détail), `error`, `result` et `status`.
  La gateway émet ses propres étapes autour des appels réels à `/predict_all` et relaie les
  événements émis par les nœuds du graphe via `POST /analyze/stream` de l'orchestrateur. Les
  services indispensables (`api-gateway`, `audio-engine`, `agent-orchestrator`) arrêtent le
  workflow en cas d'échec ; une panne MLflow est signalée sans bloquer le rapport. L'interface
  affiche uniquement le nom et le statut de chaque service (`En attente`, `En analyse`, `OK`,
  `Erreur`, `Terminé`) avec une barre de progression calculée sur les étapes achevées ; le badge
  final reste « Terminé » seulement si aucune étape n'a échoué.
- **Tracking MLflow du workflow** : la gateway crée un run dans l'expérience `cardio-workflow`
  via l'API REST, écrit les tags `step.<service>` / `detail.<service>` et les durées mesurées,
  puis clôture le run avec `runs/update` (`FINISHED` ou `FAILED`). Le rapport final conserve
  `workflow_trace` pour permettre un contrôle a posteriori sans dépendre du frontend.
  L'interface affiche ce rapport après les cinq cartes de modèles. Les modèles restent validés sur
  des signaux PCG ; aucune validation ECG n'est revendiquée.

## 5. Jalon 6 — Tests continus

- **146 tests passed** (`pytest -q`) couvrant `tests/test_audio.py` (signal synthétique + API
  avec faux ONNX), `tests/test_api.py` (proxys gateway avec `httpx` factice),
  `tests/test_agents.py` (graphe, garde-fous FR/EN, audit) et `tests/test_rag.py` (chunking,
  Qdrant mémoire, évaluation).
- Couverture (`reports/coverage.txt`) : **59 % globale**, 80–100 % sur les services applicatifs.
  Le reste correspond aux utilitaires d'entraînement (`registry.py`, `fold_utils.py`,
  `mlflow_utils.py`, `src/dataset.py`) qui ne sont pas exécutés par les tests de service.
- Aucun test ne charge de modèle volumineux, n'appelle un service distant ni ne modifie les
  notebooks ou l'audio-engine.

## 6. Jalon 7 — Architecture Docker

Six micro-services sur le réseau `docker-network` :

| Service | Image de base | Port hôte | Dépendances | Volume persistant |
|---|---|---:|---|---|
| `api-gateway` | `python:3.10-slim` (multi-stage) | 8085 | audio-engine, agent-orchestrator | — |
| `audio-engine` | `python:3.10-slim` (multi-stage) | interne 8001 | mlflow-server | `./audio_engine/models:ro` |
| `agent-orchestrator` | `python:3.10-slim` (multi-stage) | interne 8002 | llm-server, vector-db, mlflow-server | `hf_cache` |
| `llm-server` | `ollama/ollama` | 11434 | — | `ollama_data`, `./llm_finetuning/export:ro` |
| `vector-db` | `qdrant/qdrant` | 6333 | — | `qdrant_data` |
| `mlflow-server` | `python:3.10-slim` | 5000 | — | `mlflow_data` |

Commande de démarrage : `docker compose up -d --build`, puis `docker compose ps`.
Le build multi-stage installe les dépendances dans un étage `builder` et ne copie que
`/install` dans l'image finale. L'orchestrateur utilise PyTorch **CPU** et le cache Hugging Face
persistant pour `bge-m3` (premier appel ~33–37 s, appels suivants ~1–2 s). Voir le §9 du README
pour les commandes de vérification et la note sur le GPU NVIDIA.

## 7. Jalon 8 — Pipeline CI/CD

`.github/workflows/mlops-ci-cd.yml` (push / PR sur `main`, `workflow_dispatch`) :

1. job `tests` : Python 3.10, installation des dépendances, `pytest --cov` avec seuil minimal
   de couverture, archivage de `reports/coverage.xml` ;
2. job `docker` : `docker compose config --quiet`, build de l'image de l'API Gateway,
   démarrage du conteneur sur 8085 et contrôle que `/` expose `/analyze_pcg` et que `/health`
   répond `ok`.

Le pipeline ne télécharge aucun poids volumineux et ne nécessite ni GPU ni secret GitHub.

## 8. Consommation mémoire / VRAM

| Composant | Mesure / ordre de grandeur observé |
|---|---|
| Poste de travail | Windows, **RTX 2050 — 4 Go de VRAM** |
| QLoRA 1.5B | 4-bit sur GPU, fusion sur CPU (fp32 → fp16 avant sauvegarde) |
| QLoRA 7B | non exécuté : nécessite ≥ 12 Go de VRAM (GPU externes/outils type Colab) |
| GGUF livré | 940,4 Mo sur disque ; chargement dans Ollama au premier appel (~43 s) |
| ONNX audio | 44,7 Mo ; inférence CPU (`CPUExecutionProvider`) |
| `bge-m3` | chargement CPU au premier appel RAG (~33–37 s), puis mis en cache dans `hf_cache` |
| Qdrant | 66 passages vectoriels 1 024 dims (~0,3 Mo de vecteurs) |

## 9. Licences et limites

- Article : CC BY-NC 4.0. Cours universitaire : droits non explicités, **usage académique
  interne uniquement**, non redistribué.
- Les jeux de données audio appartiennent à leurs auteurs respectifs (PhysioNet, PASCAL) et
  sont soumis à leurs propres licences.
- Le fichier `.gguf`, les caches Hugging Face et les artefacts volumineux sont volontairement
  exclus de Git ; `README.md` §16 indique comment les régénérer.
- **Ce logiciel n'est pas un dispositif médical.** Il ne doit ni poser de diagnostic ni orienter
  une prise en charge clinique.
