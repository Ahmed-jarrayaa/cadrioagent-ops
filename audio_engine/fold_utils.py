"""Utilitaires partagés par les trois entraînements du TP3.

Le module centralise :

- le découpage **3 folds au niveau de l'enregistrement** (les segments d'un même
  WAV restent dans le même pli, pour éviter la fuite de données) ;
- le calcul d'une courbe d'apprentissage ``scikit-learn`` (score et perte) ;
- les figures/CSV de courbes pour les modèles à epochs (CNN TensorFlow/Keras).

Les notebooks ``SVM.ipynb``, ``Decisiontree.ipynb`` et ``CNN.ipynb`` utilisent
exactement les mêmes plis ``KFold(3, shuffle=True, random_state=10)``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, learning_curve


# --------------------------------------------------------------------------- plis
def build_3fold_recording_splits(keys_all, n_splits: int = 3, seed: int = 10):
    """Construit les plis ``train_kfoldK`` / ``test_kfoldK`` au niveau enregistrement.

    Args:
        keys_all: tableau de clés ``"classe/nom.wav"``, une entrée par segment.
        n_splits: nombre de plis (3 par défaut).
        seed: graine du ``KFold(shuffle=True)`` — identique aux notebooks SVM/arbre.

    Returns:
        ``(folds, recording_keys)`` où chaque fold est un dict avec les clés et les
        positions (indices de segments) d'entraînement et de test.
    """
    keys_all = np.asarray(keys_all)
    recording_keys = np.array(sorted(np.unique(keys_all)))

    normal = np.array([k for k in recording_keys if str(k).startswith("0/")])
    abnormal = np.array([k for k in recording_keys if str(k).startswith("1/")])
    if len(normal) == 0 or len(abnormal) == 0:
        raise ValueError("Les clés doivent contenir les classes 0/ (normal) et 1/ (abnormal)")

    positions_par_cle: dict[str, list[int]] = {}
    for position, key in enumerate(keys_all):
        positions_par_cle.setdefault(str(key), []).append(position)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, ((tr_n, te_n), (tr_a, te_a)) in enumerate(
            zip(kf.split(normal), kf.split(abnormal)), start=1):
        train_keys = set(normal[tr_n]) | set(abnormal[tr_a])
        test_keys = set(normal[te_n]) | set(abnormal[te_a])
        train_pos = np.array(sorted(p for key in train_keys for p in positions_par_cle[key]),
                             dtype=np.int64)
        test_pos = np.array(sorted(p for key in test_keys for p in positions_par_cle[key]),
                            dtype=np.int64)
        folds.append({
            "fold": fold,
            "train_keys": train_keys,
            "test_keys": test_keys,
            "train_pos": train_pos,
            "test_pos": test_pos,
        })

    return folds, recording_keys


# --------------------------------------------------------------- courbe d'apprentissage
def compute_learning_curve(estimator, X, y, groups, *, train_sizes=None,
                           cv_splits: int = 3, scoring: str = "balanced_accuracy",
                           n_jobs: int = -1, random_state: int = 10) -> dict:
    """Courbe d'apprentissage (train vs validation interne) pour un modèle sklearn.

    La validation croisée est **groupée par enregistrement** : un WAV ne peut pas
    être simultanément dans le sous-ensemble d'entraînement et de validation.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    sizes, train_scores, val_scores = learning_curve(
        clone(estimator), X, y, groups=groups,
        cv=GroupKFold(n_splits=cv_splits),
        train_sizes=train_sizes, scoring=scoring,
        n_jobs=n_jobs, shuffle=True, random_state=random_state,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    return {
        "train_sizes": sizes.astype(int).tolist(),
        "train_score_mean": train_mean.astype(float).tolist(),
        "train_score_std": train_scores.std(axis=1).astype(float).tolist(),
        "val_score_mean": val_mean.astype(float).tolist(),
        "val_score_std": val_scores.std(axis=1).astype(float).tolist(),
        # « loss » pour les modèles sans descente de gradient : 1 - score
        "train_loss_mean": (1.0 - train_mean).astype(float).tolist(),
        "val_loss_mean": (1.0 - val_mean).astype(float).tolist(),
        "scoring": scoring,
    }


def save_learning_curve_csv(curve: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["train_sizes", "train_score_mean", "train_score_std",
            "val_score_mean", "val_score_std", "train_loss_mean", "val_loss_mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in zip(*(curve[key] for key in keys)):
            writer.writerow(row)
    return path


def plot_learning_curve(curve: dict, fold: int, out_dir: Path, model_name: str,
                        color: str = "#2a7ae2") -> Path:
    """Figure « score » + « perte » d'un pli ; renvoie le chemin du PNG."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = np.asarray(curve["train_sizes"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key_mean, key_std, ylabel, title in (
        (axes[0], "train_score_mean", "train_score_std", "score", "Courbe d'apprentissage"),
        (axes[1], "train_loss_mean", None, "perte (1 - score)", "Courbe de perte"),
    ):
        train_mean = np.asarray(curve[key_mean])
        val_key = key_mean.replace("train_", "val_")
        val_mean = np.asarray(curve[val_key])
        ax.plot(sizes, train_mean, marker="o", color=color, label="entraînement")
        ax.plot(sizes, val_mean, marker="s", color="#c2414b", label="validation (CV groupée)")
        if key_std is not None:
            train_std = np.asarray(curve[key_std])
            val_std = np.asarray(curve[key_std.replace("train_", "val_")])
            ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                            color=color, alpha=0.15)
            ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                            color="#c2414b", alpha=0.15)
        ax.set_xlabel("nombre de segments d'entraînement")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {model_name.upper()} — pli {fold}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    path = out_dir / f"{model_name}_fold{fold}_learning_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all_folds_learning_curves(curves: list[dict], out_dir: Path, model_name: str) -> Path:
    """Superpose les courbes des 3 plis (score et perte)."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = ["#2a7ae2", "#e08a2a", "#2e9d63"]

    for i, curve in enumerate(curves):
        sizes = np.asarray(curve["train_sizes"])
        color = palette[i % len(palette)]
        fold = curve.get("fold", i + 1)
        axes[0].plot(sizes, curve["val_score_mean"], marker="o", color=color,
                     label=f"pli {fold} — validation")
        axes[0].plot(sizes, curve["train_score_mean"], marker="o", linestyle="--",
                     color=color, alpha=0.75)
        axes[1].plot(sizes, curve["val_loss_mean"], marker="s", color=color,
                     label=f"pli {fold} — validation")
        axes[1].plot(sizes, curve["train_loss_mean"], marker="s", linestyle="--",
                     color=color, alpha=0.75)

    axes[0].set_xlabel("nombre de segments d'entraînement")
    axes[0].set_ylabel("score")
    axes[0].set_title(f"Courbes d'apprentissage — {model_name.upper()} (3 plis)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xlabel("nombre de segments d'entraînement")
    axes[1].set_ylabel("perte (1 - score)")
    axes[1].set_title(f"Courbes de perte — {model_name.upper()} (3 plis)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = out_dir / f"{model_name}_learning_curves_all_folds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------- historiques (CNN Keras)
def save_history_csv(history: dict, path: Path) -> Path:
    """Sauvegarde un ``history.history`` Keras en CSV (une ligne par epoch)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history)
    n_rows = len(history[keys[0]]) if keys else 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for row in range(n_rows):
            writer.writerow([row + 1, *(history[key][row] for key in keys)])
    return path


def plot_training_history(history: dict, fold: int, out_dir: Path, model_name: str) -> Path:
    """Figure loss + accuracy/AUC d'un pli CNN ; renvoie le chemin du PNG."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"Perte — pli {fold}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if "accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], marker="o", label="train accuracy")
        axes[1].plot(epochs, history["val_accuracy"], marker="s", label="val accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"Exactitude — pli {fold}")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if "val_auc" in history:
        axes[2].plot(epochs, history["val_auc"], marker="d", color="#2e9d63",
                     label="val AUC")
        axes[2].set_ylabel("AUC")
    axes[2].set_xlabel("epoch")
    axes[2].set_title(f"AUC validation — pli {fold}")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    path = out_dir / f"{model_name}_fold{fold}_training.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all_folds_training(histories: list[dict], out_dir: Path, model_name: str) -> Path:
    """Superpose loss et AUC de validation des 3 plis CNN."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = ["#2a7ae2", "#e08a2a", "#2e9d63"]

    for i, entry in enumerate(histories):
        history = entry.get("metrics", entry)
        fold = entry.get("fold", i + 1)
        color = palette[i % len(palette)]
        epochs = np.arange(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], marker="o", color=color,
                     label=f"pli {fold} — train")
        axes[0].plot(epochs, history["val_loss"], marker="o", linestyle="--",
                     color=color, label=f"pli {fold} — val")
        if "val_auc" in history:
            axes[1].plot(epochs, history["val_auc"], marker="s", color=color,
                         label=f"pli {fold}")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss — {model_name.upper()} (3 plis)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("AUC validation")
    axes[1].set_title(f"AUC validation — {model_name.upper()} (3 plis)")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / f"{model_name}_training_all_folds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
