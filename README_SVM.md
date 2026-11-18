# Pipeline SVM - CardioAgent-Ops

Depuis la racine du projet (`Data Mining/`), environnement virtuel active (`.\.venv\Scripts\Activate.ps1`),
MLflow demarre (`docker compose up -d mlflow-server`).

| Etape | Commande | Resultat |
|---|---|---|
| 0. Dependances | `python -m pip install -r requirements-train.txt` | |
| 1. Donnees | `python audio_engine/prepare_data.py` | `ourdataset/processed/dataset_2000hz.npz` |
| 2. Entrainement | `python audio_engine/train_svm.py` | modele ONNX + metriques + run MLflow |
| 3. Graphiques | `python audio_engine/make_plots.py` | `audio_engine/models/plots/*.png` |
| 4. Verification | `python audio_engine/check_onnx.py` | ecart ONNX vs scikit-learn (~1e-6) |
| 5. Service | `docker compose up -d --build audio-engine` | `/predict` charge le modele |

Les etapes 2, 3 et 4 sont des scripts separes a dessein : sous Windows, charger matplotlib, onnx et
onnxruntime dans le meme processus peut provoquer un plantage de DLL (access violation).

## Fichiers
- `preprocess.py` : pretraitement (2000 Hz, filtre 25-400 Hz, segments de 5 s, MFCC/ZCR/centroide/RMS).
  Utilise par `prepare_data.py` ET par `server.py` : jamais de divergence entre entrainement et inference.
- `prepare_data.py` : inventaire, doublons, fichiers illisibles/trop courts/silencieux, split 70/15/15 par enregistrement.
- `train_svm.py` : SVM RBF, seuil regle sur la validation (Recall max avec specificite >= 0,8), MLflow, export ONNX.
- `make_plots.py`, `check_onnx.py` : graphiques et verification.
- `server.py` : service FastAPI d'inference (conteneur audio-engine).
