# audio_engine/src/dataset.py
"""Préparation des Mel-spectrogrammes et ``Sequence`` Keras pour TensorFlow.

Le cache Mel est calculé une seule fois puis consommé par un
``tf.keras.utils.Sequence`` : l'entraînement ne charge que les lots nécessaires,
ce qui évite de copier tout le cache de spectrogrammes en RAM à chaque epoch.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import tensorflow as tf


# --------- Pré-calcul des Mel-spectrogrammes (une seule fois) ---------
def precompute_mels(signals, sr=2000, n_mels=128, n_fft=512,
                    hop_length=128, target_time_frames=256, verbose=True):
    """Transforme un batch de signaux bruts (N, T) en Mel-dB (N, n_mels, T)."""
    mels = np.zeros((len(signals), n_mels, target_time_frames), dtype=np.float32)

    for i, sig in enumerate(signals):
        if sig.ndim > 1:
            sig = sig.flatten()
        sig = sig.astype(np.float32)

        m = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels,
            fmin=20, fmax=sr // 2,
        )
        m = librosa.power_to_db(m, ref=np.max)

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


# --------- Sequence Keras (utilise le cache memmap) ---------
class KerasMelSequence(tf.keras.utils.Sequence):
    """Sequence Keras qui consomme les Mel déjà calculés + un sous-ensemble d'indices.

    Args:
        mels: array (N, 1, n_mels, T) ou (N, n_mels, T), éventuellement memmap.
        labels: array (N,) labels binaires 0/1.
        indices: indices de segments du split (train ou validation).
        batch_size: taille de lot.
        shuffle: mélange les indices à chaque epoch (train uniquement).
        augment: applique un SpecAugment léger (train uniquement).
        seed: graine du générateur d'augmentation.
    """

    def __init__(self, mels, labels, indices, batch_size=32, shuffle=False,
                 augment=False, seed=42):
        super().__init__()
        self.mels = mels
        self.labels = np.asarray(labels)
        self.indices = np.array(indices, dtype=np.int64, copy=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self._has_channel_axis = (np.asarray(mels.shape).size == 4)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        batch_pos = self.indices[start:stop]

        # float16 (cache) -> float32 (TensorFlow), puis canaux en dernier.
        x = np.asarray(self.mels[batch_pos], dtype=np.float32)
        if self._has_channel_axis:
            x = np.transpose(x, (0, 2, 3, 1))       # (B, 1, 128, 256) -> (B, 128, 256, 1)
        else:
            x = x[..., None]

        if self.augment:
            for i in range(x.shape[0]):
                x[i, ..., 0] = self._spec_augment(x[i, ..., 0])

        y = self.labels[batch_pos].astype(np.int64)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _spec_augment(self, spec):
        """SpecAugment léger : masque temporel + masque fréquentiel."""
        spec = spec.copy()
        n_mels, t = spec.shape

        if self.rng.random() < 0.5:
            width = int(self.rng.integers(0, max(1, t // 10)))
            if width > 0:
                start = int(self.rng.integers(0, t - width))
                spec[:, start:start + width] = 0.0

        if self.rng.random() < 0.5:
            height = int(self.rng.integers(0, max(1, n_mels // 10)))
            if height > 0:
                start = int(self.rng.integers(0, n_mels - height))
                spec[start:start + height, :] = 0.0

        return spec
