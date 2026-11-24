"""Prétraitement audio de l'audio-engine.

DOIT rester strictement identique au notebook 01_eda_data_wrangling.ipynb (sections 5 et 6) :
sinon le modèle ONNX reçoit des entrées qui n'ont pas la distribution de l'entraînement.
"""

import io
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_CFG = {
    "sample_rate": 2000, "bandpass_hz": [25, 400], "segment_sec": 5,
    "n_fft": 512, "hop_length": 64, "n_mels": 40, "n_mfcc": 20,
    "min_duration_s": 1.0, "max_segments": 20,
}


class AudioError(ValueError):
    """Fichier audio inutilisable (illisible, trop court, silencieux)."""


def feature_names(n_mfcc=20):
    return ([f"mfcc{i}_mean" for i in range(n_mfcc)] + [f"mfcc{i}_std" for i in range(n_mfcc)] +
            ["zcr_mean", "zcr_std", "centroid_mean", "centroid_std", "rms_mean", "rms_std"])


class Preprocessor:
    def __init__(self, cfg=None):
        c = {**DEFAULT_CFG, **{k: v for k, v in (cfg or {}).items() if k in DEFAULT_CFG or k == "hop_sec"}}
        self.sr = int(c["sample_rate"])
        self.seg_len = int(c["segment_sec"] * self.sr)
        self.hop_len = int(c.get("hop_sec", c["segment_sec"] / 2) * self.sr)   # 50 % de chevauchement
        self.n_fft, self.hop = int(c["n_fft"]), int(c["hop_length"])
        self.n_mels, self.n_mfcc = int(c["n_mels"]), int(c["n_mfcc"])
        self.min_dur, self.max_seg = float(c["min_duration_s"]), int(c["max_segments"])
        self.sos = butter(4, c["bandpass_hz"], btype="band", fs=self.sr, output="sos")
        self.bandpass_hz = [float(v) for v in c["bandpass_hz"]]
        self.names = feature_names(self.n_mfcc)

    # ---- signal ----
    def load_clean(self, data: bytes):
        try:
            y, _ = librosa.load(io.BytesIO(data), sr=self.sr, mono=True)   # rééchantillonnage à 2000 Hz
        except Exception as e:
            raise AudioError("Fichier audio illisible (WAV attendu)") from e
        if len(y) / self.sr < self.min_dur:
            raise AudioError(f"Enregistrement trop court (< {self.min_dur} s)")
        if np.max(np.abs(y)) < 1e-6:
            raise AudioError("Signal vide (silence)")
        y = y - np.mean(y)
        y = sosfiltfilt(self.sos, y)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y.astype(np.float32)

    def segment_positions(self, y):
        """Indices de début des fenêtres de ``segment`` (pour la visualisation)."""
        if len(y) < self.seg_len:
            return [0]
        starts = np.arange(0, len(y) - self.seg_len + 1, self.hop_len)
        if len(starts) > self.max_seg:        # fenêtres réparties sur tout l'enregistrement
            starts = starts[np.linspace(0, len(starts) - 1, self.max_seg).round().astype(int)]
        return [int(i) for i in starts]

    def segment(self, y):
        if len(y) < self.seg_len:
            return [np.resize(y, self.seg_len)]
        return [y[i:i + self.seg_len] for i in self.segment_positions(y)]


    # ---- caractéristiques ----
    def features_1d(self, s):
        mfcc = librosa.feature.mfcc(y=s, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                    hop_length=self.hop, n_mels=self.n_mels)
        zcr = librosa.feature.zero_crossing_rate(s, frame_length=self.n_fft, hop_length=self.hop)
        cent = librosa.feature.spectral_centroid(y=s, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop)
        rms = librosa.feature.rms(y=s, frame_length=self.n_fft, hop_length=self.hop)
        parts = [mfcc.mean(1), mfcc.std(1)]
        for m in (zcr, cent, rms):
            parts += [np.array([m.mean()]), np.array([m.std()])]
        return np.concatenate(parts).astype(np.float32)

    def mel_db(self, s):
        m = librosa.feature.melspectrogram(y=s, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop,
                                           n_mels=self.n_mels)
        # float16 puis float32 : reproduit exactement le stockage du dataset d'entraînement
        return librosa.power_to_db(m, ref=np.max).astype(np.float16).astype(np.float32)

    def batch(self, segments, kind):
        if kind == "features":
            return np.stack([self.features_1d(s) for s in segments]).astype(np.float32)
        return np.stack([self.mel_db(s) for s in segments])[:, None, :, :].astype(np.float32)
