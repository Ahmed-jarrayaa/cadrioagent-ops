# Pipeline SVM — note complémentaire

Ce document complète le [README principal](README.md). Le pipeline SVM est désormais entraîné
dans le notebook **`audio_engine/SVM.ipynb`** ; les anciens scripts `prepare_data.py`,
`train_svm.py`, `make_plots.py` et `check_onnx.py` ne font plus partie du dépôt.

## Démarrage rapide

Depuis la racine du projet, environnement virtuel activé
(`.\.venv\Scripts\Activate.ps1`) :

| Étape | Commande | Résultat |
|---|---|---|
| 0. Dépendances | `python -m pip install -r requirements-train.txt` | |
| 1. MLflow (optionnel) | `docker compose up -d mlflow-server` | serveur MLflow sur `:5000` |
| 2. Entraînement | exécuter `audio_engine/SVM.ipynb` | `audio_engine/models/best_svm.joblib` + `best_svm_meta.json` + run MLflow |
| 3. Vérification | `cd audio_engine` puis `python analyze_signal.py ..\ourdataset\train\abnormal\112_1306243000964_A.wav --families svm` | prédiction SVM |
| 4. Service | `python -m uvicorn server:app --host 127.0.0.1 --port 8001` (depuis `audio_engine/`) | API `/predict_one?family=svm` |

## Fichiers actuels

- `audio_engine/preprocess.py` : prétraitement commun (2 000 Hz, filtre 25–400 Hz, segments de
  5 s, 46 caractéristiques MFCC/ZCR/centroïde/RMS). Utilisé à l'entraînement **et** à l'inférence.
- `audio_engine/SVM.ipynb` : `GridSearchCV` + `GroupKFold(3)`, seuil réglé sur la validation
  (rappel maximal avec spécificité ≥ 0,80), suivi MLflow.
- `audio_engine/registry.py` : chargement du modèle, prédiction par segment et agrégation.
- `audio_engine/models/best_svm.joblib` + `best_svm_meta.json` : artefacts servis par l'API.
- `audio_engine/models/plots/` : matrices de confusion, courbes ROC et comparaisons.

> Sous Windows, éviter de charger `matplotlib`, `onnx` et `onnxruntime` dans le même processus :
> exécuter les étapes d'entraînement, de tracé et de vérification séparément.
