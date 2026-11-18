# audio_engine/src/dataset.py
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


# --------- Pré-calcul des Mel-spectrogrammes (une seule fois) ---------
def precompute_mels(signals, sr=2000, n_mels=128, n_fft=512,
                    hop_length=128, target_time_frames=256, verbose=True):
    """
    Transforme un batch de signaux bruts (N, T) en Mel-dB (N, n_mels, target_time_frames).
    Sauvegarde automatique dans un cache pour éviter le recalcul.
    """
    mels = np.zeros((len(signals), n_mels, target_time_frames), dtype=np.float32)

    for i, sig in enumerate(signals):
        if sig.ndim > 1:
            sig = sig.flatten()
        sig = sig.astype(np.float32)

        # Mel spectrogram → dB
        m = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels,
            fmin=20, fmax=sr // 2,
        )
        m = librosa.power_to_db(m, ref=np.max)

        # Pad / crop temporel
        t = m.shape[1]
        if t < target_time_frames:
            m = np.pad(m, ((0, 0), (0, target_time_frames - t)))
        else:
            m = m[:, :target_time_frames]

        mels[i] = m

        if verbose and (i + 1) % 100 == 0:
            print(f"  [Mel] {i+1}/{len(signals)}")

    return mels


def load_or_compute_mels(signals, cache_path, **kwargs):
    """Charge depuis le cache si dispo, sinon calcule et sauvegarde."""
    if os.path.exists(cache_path):
        print(f"[INFO] Chargement cache Mel : {cache_path}")
        return np.load(cache_path)
    print(f"[INFO] Calcul des Mel-spectrogrammes ({len(signals)} signaux)...")
    mels = precompute_mels(signals, **kwargs)
    np.save(cache_path, mels)
    print(f"[INFO] Cache sauvegardé : {cache_path}")
    return mels


# --------- Dataset PyTorch (utilise le cache) ---------
class CachedMelDataset(Dataset):
    """
    Dataset qui consomme les Mel déjà calculés + un sous-ensemble d'indices
    (pour gérer les folds train/val).
    """
    def __init__(self, mels, labels, indices, augment=False):
        """
        mels    : array (N, n_mels, T) déjà pré-calculé
        labels  : array (N,) labels binaires 0/1
        indices : liste/array des indices à utiliser pour ce split
        """
        self.mels = mels
        self.labels = labels
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _spec_augment(self, spec):
        """SpecAugment léger : masque temporel + fréquentiel."""
        spec = spec.copy()
        n_mels, t = spec.shape

        if np.random.rand() < 0.5:
            f = np.random.randint(0, max(1, t // 10))
            t0 = np.random.randint(0, t - f)
            spec[:, t0:t0 + f] = 0.0

        if np.random.rand() < 0.5:
            f = np.random.randint(0, max(1, n_mels // 10))
            f0 = np.random.randint(0, n_mels - f)
            spec[f0:f0 + f, :] = 0.0

        return spec

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        spec = self.mels[real_idx]

        # Normalisation par instance (z-score)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if self.augment:
            spec = self._spec_augment(spec)

        # (1, n_mels, T)
        spec = np.expand_dims(spec, axis=0)
        return (torch.tensor(spec, dtype=torch.float32),
                torch.tensor(int(self.labels[real_idx]), dtype=torch.long))