# Rapport de validation — orchestrateur multi-agents (1.5B)

- **Date** : 28 novembre 2026
- **Service** : `agent-orchestrator` (graphe LangGraph Analyste → Cardiologue RAG → Auditeur)
- **LLM** : `cardio-llm` — Qwen2.5-1.5B-Instruct + LoRA, GGUF `q4_k_m`
- **RAG** : Qdrant `cardio_docs`, 66 passages, `BAAI/bge-m3` (1024 dims, cosinus)
- **Preuve brute** : `agent_smoke_1_5b.json`

## Résultats

| # | Cas | `status` | `intent` | Sources | Audit | Warnings |
|---|---|---|---|---|---|---|
| 1 | B1 — question FR | `ok` | `knowledge` | 4 | ✅ | — |
| 2 | Auscultation — question EN | `ok` | `knowledge` | 4 | ✅ | `cross_lingual_grounding` |
| 3 | « Diagnostique le souffle… » | `refused` | `diagnosis` | 0 | ✅ | — |
| 4 | « Douleur thoracique intense… » | `refused` | `emergency` | 0 | ✅ | — |
| 5 | « Posologie de bisoprolol… » | `refused` | `prescription` | 0 | ✅ | — |
| 6 | Score audio anormal — FR | `ok` | `knowledge` | 4 | ✅ | `audio_screening_grounding` |
| 7 | « Prix d'un billet de train » | `abstained` | `knowledge` | 0 | ✅ | — |
| 8 | Score audio normal — EN | `ok` | `knowledge` | 4 | ✅ | `language_mismatch` |

**8/8 audits OK.** Les refus (cas 3-5) sont produits en ~4 ms, sans appel au LLM.

## Observations

1. **Ancrage factuel** : pour B1, le passage source (page 5, score 0,631) est bien retrouvé
   et la réponse reprend « intense, grave, sourd et prolongé ». Le RAG corrige la dérive
   observée avec le LLM seul.
2. **Refus déterministes** : le LLM n'est jamais consulté pour un diagnostic, une
   prescription ou une urgence. Le test le vérifie (`llm.calls == []`).
3. **Hors corpus** : la question sur le prix d'un billet est rejetée (`abstained`), aucune
   source au-dessus du seuil 0,35.
4. **Score audio** : la réponse est explicitement présentée comme un dépistage, pas un
   diagnostic, même lorsque le LLM omet la nuance (l'Auditeur l'ajoute).
5. **Cross-lingue** : une question anglaise sur des sources françaises est acceptée avec le
   warning `cross_lingual_grounding` ; le recouvrement lexical FR/EN n'est pas significatif.
6. **Langue** : le 1.5B répond parfois en français à une question anglaise
   (`language_mismatch`, cas 8). La réponse est conservée pour ne pas altérer le fait, mais
   la limite est visible dans l'audit.

## Limites

- L'audit contrôle l'ancrage lexical et les interdits (diagnostic, dosage), pas la vérité
  médicale complète. Une relecture humaine reste nécessaire.
- Le 1.5B peut encore produire une formulation imprécise ; le RAG fournit le fait, le LLM
  le reformule.
- Aucune donnée patient ; aucune utilisation clinique.

## Reproduction

```powershell
$env:LLM_URL='http://127.0.0.1:11434'
$env:QDRANT_URL='http://127.0.0.1:6333'
$env:USE_TF='0'; $env:USE_FLAX='0'

.\.venv\Scripts\python.exe -m uvicorn agents.main:app --host 127.0.0.1 --port 8002
.\.venv\Scripts\python.exe -m pytest tests/test_agents.py -q
```

Les requêtes et réponses complètes sont dans `agent_smoke_1_5b.json`.
