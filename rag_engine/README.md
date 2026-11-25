# Jalon 4 — Pipeline RAG & base vectorielle

Le moteur RAG fournit les **faits extraits des deux PDF du projet** à l'orchestrateur.
Le LLM fine-tuné (`cardio-llm`, §8 du README principal) reformule ces faits : il n'est jamais
la source unique d'une affirmation médicale. C'est la mitigation directe de l'hallucination
observée sur B1 (« sifflement aigu » au lieu de « bruit sourd, grave, intense »).

> Usage pédagogique uniquement. Ni diagnostic, ni prescription, ni décision clinique.

## Sources (aucune donnée inventée)

| Identifiant | Fichier | Licence |
|---|---|---|
| `auscultation_cardiaque_normale` | `Auscultation-cardiaque-normale-4-4.pdf` | Droits non explicités — usage académique interne |
| `esc_evaluation_cardiovasculaire` | `S127979602300089X.pdf` | CC BY-NC 4.0 (attribution, non commercial) |

Les empreintes SHA-256 et les métadonnées sont déclarées dans
`llm_finetuning/data/sources.json`. Chaque passage indexé porte `source`, `source_id`,
`page`, `chunk_index`, `license` et un `disclaimer` : toute réponse peut être retracée.

## Composants

| Fichier | Rôle |
|---|---|
| `chunk.py` | Extraction `pypdf`, normalisation, découpage (900 caractères, recouvrement 120) et IDs SHA-256 stables |
| `embed.py` | Embeddings `BAAI/bge-m3` (FR/EN, 1024 dims), normalisés L2, cache `.npz` |
| `index_qdrant.py` | Création de la collection `cardio_docs` (distance cosinus) + upsert idempotent |
| `retrieve.py` | Recherche par similarité, formatage du contexte avec provenance, refuses si aucun passage |
| `evaluate.py` | Évaluation reproductible du rappel (6 questions) |
| `eval_queries.json` | Questions, seuils et attentes de l'évaluation |
| `reports/rag_evaluation.json` | Rapport généré (preuve versionnée) |
| `requirements.txt` | Dépendances Python du pipeline |

## Utilisation

Prérequis : service `vector-db` démarré (`docker compose up -d vector-db`) et dépendances
installées dans `.venv`.

```powershell
$env:USE_TF='0'; $env:USE_FLAX='0'

# Indexation complète (66 passages, ~1 min en CPU)
.\.venv\Scripts\python.exe -m rag_engine.index_qdrant --recreate

# Évaluation (6/6 attendu)
.\.venv\Scripts\python.exe -m rag_engine.evaluate

# Recherche manuelle
.\.venv\Scripts\python.exe -m rag_engine.retrieve "Que disent les recommandations ESC ?" --top-k 3
```

Variables d'environnement : `QDRANT_URL` (défaut `http://127.0.0.1:6333`,
`http://vector-db:6333` dans Docker), `QDRANT_COLLECTION` (défaut `cardio_docs`),
`HF_HUB_OFFLINE=1` pour interdire tout téléchargement de modèle.

Le mode `--in-memory` de `index_qdrant.py` et les tests `tests/test_rag.py` fonctionnent sans
Docker ni modèle Hugging Face (encodeur factice).

## Résultats mesurés

- **66 passages** indexés (2 sources), 1024 dimensions, distance cosinus.
- **6/6 questions validées** : B1 en FR et EN, recommandations ESC, rôle de l'auscultation,
  présence du disclaimer, et **rejet d'une question hors corpus** (score < 0,35).
- Exemple : « Qu'est-ce que le premier bruit du coeur B1 ? » → score **0,631**, page 5 du
  cours : « *Le B1 est intense, grave, sourd et prolongé…* ». Le passage exact est ainsi
  fourni au LLM au lieu d'une paraphrase inventée.

Seuil recommandé : `min_score = 0,35`. En dessous, `format_context()` renvoie
`Aucun passage pertinent trouvé dans les documents sources.` — le LLM doit alors le dire
clairement au lieu d'inventer.

## Intégration suivante (Jalon 5)

L'orchestrateur appellera :

```python
from rag_engine.retrieve import format_context, retrieve

passages = retrieve(question, top_k=4, min_score=0.35)
context = format_context(passages)
```

Le contexte est injecté dans le prompt de l'agent « Cardiologue RAG », puis l'agent
« Auditeur » vérifie que chaque affirmation est appuyée par une source. Les garde-fous
déjà validés dans le `Modelfile` restent actifs.
