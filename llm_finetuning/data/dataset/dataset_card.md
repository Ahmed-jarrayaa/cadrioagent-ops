# Dataset Card — CardioAgent LLM (fr/en)

> ⚠️ Corpus **pédagogique**. Il ne constitue ni un dispositif médical ni une aide au
> diagnostic. Ne pas l'utiliser pour une décision clinique.

## Résumé

- Version : `1.0.0`
- Total d'exemples : **590**
- Format : JSONL, champs `instruction`, `input`, `output` (+ métadonnées `meta`).
- Langues : français et anglais.
- Tâches : question/réponse, explication au patient, question inverse, guardrails.

## Répartition

| Split | Total | FR | EN | Fiches uniques |
|---|---:|---:|---:|---:|
| train | 450 | 225 | 225 | 110 |
| validation | 70 | 35 | 35 | 15 |
| test | 70 | 35 | 35 | 15 |

## Sources

| ID | Titre | Licence | Fichier |
|---|---|---|---|
| `auscultation_cardiaque_normale` | Auscultation cardiaque normale | Droits non explicitement précisés — usage académique interne | `rag_engine/Auscultation-cardiaque-normale-4-4.pdf` |
| `esc_evaluation_cardiovasculaire` | Les nouvelles recommandations de la Société européenne de cardiologie sur l'évaluation cardiovasculaire changent-elles ma pratique ? | CC BY-NC 4.0 | `rag_engine/S127979602300089X.pdf` |

## Méthode de construction

1. Extraction du texte des deux PDF fournis (`pdftotext -layout`).
2. Rédaction manuelle de **90 fiches de faits** ancrées, avec extrait justificatif.
3. Génération déterministe de 6 tâches par fiche (FR/EN).
4. Ajout de 50 guardrails de sécurité (FR/EN), sans donnée médicale nouvelle.
5. Découpage stratifié par source et par fiche : toutes les variantes d'une fiche
   restent dans le même split pour éviter les fuites.

## Limites

- Le corpus couvre deux documents seulement : il est volontairement restreint.
- Les réponses sont des reformulations pédagogiques, pas des recommandations de prise
  en charge individuelles.
- Le cours d'auscultation a des droits non explicités : usage académique interne.
- L'article est sous licence CC BY-NC 4.0 : usage non commercial avec attribution.
- Aucune validation clinique n'a été réalisée.

## Reproductibilité

```bash
python llm_finetuning/data/build_dataset.py
pytest llm_finetuning/tests -q
```
