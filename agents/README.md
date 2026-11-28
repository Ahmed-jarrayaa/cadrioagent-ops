# Jalon 5 — Multi-Agents LangGraph & guardrails Pydantic

Graphe d'états à trois agents pour le service `agent-orchestrator` (port 8002) :

```text
START → Analyste → Cardiologue RAG → Auditeur → END
          │              │               │
          │              │               └─ vérifie ancrage, refus, dosage
          │              └─ Qdrant (rag_engine) + Ollama (cardio-llm)
          └─ langue, intention, requête de retrieval
```

> Usage pédagogique. Ni diagnostic, ni prescription, ni décision clinique.

## Fichiers

| Fichier | Rôle |
|---|---|
| `schemas.py` | Contrats Pydantic : `AnalyzeRequest`, `AnalyzeResponse`, `AuditReport`, `GuardrailReport` |
| `state.py` | État partagé `AnalysisState` (TypedDict) |
| `nodes.py` | Les 3 agents et les règles déterministes de sécurité |
| `graph.py` | Compilation LangGraph (`StateGraph`) |
| `service.py` | Orchestration requête → graphe → réponse validée |
| `clients.py` | `QdrantRetriever` (RAG) et `OllamaClient` (`/api/generate`) |
| `prompts.py` | Prompts FR/EN et messages de refus canoniques |
| `config.py` | Paramètres lus dans l'environnement |
| `main.py` | API FastAPI (`/health`, `/analyze`) |
| `reports/agent_smoke_1_5b.json` | Preuve du test local (8 cas) |

## Contrat d'entrée

Deux modes, dans le même endpoint `POST /analyze` :

```json
{ "question": "Qu'est-ce que le premier bruit B1 ?", "top_k": 4, "language": "auto" }
```

```json
{ "prediction": "abnormal", "prob_abnormal": 0.82, "threshold": 0.5, "model": "cardio-onnx" }
```

Le second mode accepte le payload exact de l'audio-engine (`/predict`), donc
`api-gateway /analyze_pcg` fonctionne sans adaptation. La sortie est un
`AnalyzeResponse` avec `status`, `intent`, `answer`, `sources`, `audit`, `guardrails`,
`audio` et `disclaimer`.

## Garde-fous

| Intention détectée | Comportement | Appel LLM |
|---|---|---|
| `emergency` (douleur thoracique, infarctus, urgence…) | redirection immédiate vers les secours | **non** |
| `diagnosis` (« diagnostique… », « est-ce que c'est… », « ai-je… ») | refus canonique + consultation | **non** |
| `prescription` (posologie, dose, « quel médicament ») | refus canonique + consultation | **non** |
| `knowledge` avec sources | reformulation ancrée, citations `[n]` | oui |
| `knowledge` sans source au-dessus du seuil | abstention explicite | **non** |
| `knowledge` avec LLM indisponible | passage source verbatim (repli extractif) | — |
| réponse non ancrée / diagnostic / dosage dans la réponse | remplacement par un message sûr + issue d'audit | — |

Les messages de refus sont **déterministes** : même si le LLM hallucine, il n'est jamais
appelé pour un diagnostic, une prescription ou une urgence.

Un score audio est toujours requalifié en **test de dépistage**, jamais en diagnostic. Le
`Modelfile` du LLM contient la même consigne, en défense en profondeur.

## Résultats mesurés (local, 28/11/2026)

`agents/reports/agent_smoke_1_5b.json` — Qdrant `cardio_docs` (66 passages, `BAAI/bge-m3`),
Ollama `cardio-llm` (Qwen2.5-1.5B fine-tuné).

| Cas | `status` | `intent` | Audit | Warnings |
|---|---|---|---|---|
| B1, question FR | `ok` | `knowledge` | ✅ | — |
| Auscultation, question EN | `ok` | `knowledge` | ✅ | `cross_lingual_grounding` |
| Diagnostic FR | `refused` | `diagnosis` | ✅ | — |
| Urgence FR | `refused` | `emergency` | ✅ | — |
| Prescription FR | `refused` | `prescription` | ✅ | — |
| Score audio anormal FR | `ok` | `knowledge` | ✅ | `audio_screening_grounding` |
| Question hors corpus FR | `abstained` | `knowledge` | ✅ | — |
| Score audio normal EN | `ok` | `knowledge` | ✅ | `language_mismatch` |

- **8/8 audits OK**, 3 refus déterministes en ~4 ms (sans LLM).
- Premier appel : ~33-37 s (chargement de `bge-m3` en CPU) ; appels suivants : ~1-2 s.
- Exemple B1 : source `auscultation_cardiaque_normale`, page 5, score 0,631 ; réponse
  « B1 se situe en début de systole et est intense, grave, sourd et prolongé… ».

### Warnings assumés

- `cross_lingual_grounding` : réponse anglaise sur sources françaises ; le recouvrement
  lexical n'est pas pertinent, l'ancrage est accepté mais signalé.
- `audio_screening_grounding` : l'explication d'un score audio s'appuie sur le résumé fourni,
  pas sur un recouvrement mot-à-mot avec les passages RAG.
- `language_mismatch` : le 1.5B répond parfois en français à une question anglaise. La
  réponse reste conservée ; le warning rend la limite visible.

## Limites

- L'auditeur vérifie l'**ancrage lexical**, pas la vérité médicale : le 1.5B peut encore
  reformuler une nuance de façon imprécise. Le RAG fournit le fait, mais une relecture
  humaine reste nécessaire.
- Aucune donnée patient n'est utilisée ; ne pas soumettre de données identifiantes.
- Pas de journalisation MLflow dans l'orchestrateur à ce stade (les runs audio et LLM sont
  déjà suivis séparément).
- Le 7B n'est pas exécuté ; le service est validé avec le 1.5B livré.

## Exécution locale

```powershell
$env:LLM_URL='http://127.0.0.1:11434'
$env:QDRANT_URL='http://127.0.0.1:6333'
$env:USE_TF='0'; $env:USE_FLAX='0'

.\.venv\Scripts\python.exe -m uvicorn agents.main:app --host 127.0.0.1 --port 8002

# tests (sans réseau ni modèle)
.\.venv\Scripts\python.exe -m pytest tests/test_agents.py -q
```

Variables : `LLM_URL`, `LLM_MODEL`, `QDRANT_URL`, `QDRANT_COLLECTION`,
`RAG_EMBEDDING_MODEL`, `RAG_MIN_SCORE`, `RAG_TOP_K`, `RAG_MAX_CONTEXT_CHARS`,
`LLM_TIMEOUT_S`, `LLM_TEMPERATURE`, `LLM_NUM_PREDICT`.

## Docker (Jalon 7)

Le `Dockerfile` copie `agents/` et `rag_engine/`, installe PyTorch **CPU** et lance
`uvicorn agents.main:app`. Le volume nommé `hf_cache` conserve `BAAI/bge-m3` entre les
redémarrages ; le premier appel télécharge le modèle (~2,2 Go) si le cache est vide.

```powershell
docker compose build agent-orchestrator
docker compose up -d --force-recreate agent-orchestrator
docker compose logs -f agent-orchestrator
```

Le port 8002 reste interne : passer par `api-gateway` (`:8085/analyze`) ou publier le port
pour un test local.
