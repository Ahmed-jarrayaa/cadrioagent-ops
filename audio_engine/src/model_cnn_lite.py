"""Architecture CNN « lite » pour la classification PCG, en TensorFlow/Keras.

Le modèle est volontairement petit (≈ 421 k paramètres) pour tourner sur CPU.
Il est importable par le notebook d'entraînement et par le service d'inférence
(``registry.py``), et sérialisable en ``.keras``.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def _conv_block(x, filters: int, dropout: float = 0.0):
    """Conv2D → BatchNorm → ReLU → MaxPool (+ Dropout optionnel)."""
    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_cnn_lite(n_classes: int = 2, dropout: float = 0.3,
                   input_shape: tuple[int, int, int] = (64, 128, 1),
                   name: str = "cardio_cnn_lite") -> keras.Model:
    """Construit le CNN Keras.

    Args:
        n_classes: nombre de classes (2 : normal / abnormal).
        dropout: dropout de la couche fully-connected finale.
        input_shape: forme d'une entrée Mel, par défaut ``(64, 128, 1)``.
        name: nom du modèle Keras.

    Returns:
        Un ``keras.Model`` qui produit des probabilités (softmax).
    """
    inputs = keras.Input(shape=input_shape, name="mel_input")

    x = _conv_block(inputs, 32, dropout=0.0)    # → (B, 64, 128, 32)
    x = _conv_block(x, 64, dropout=0.1)         # → (B, 32,  64, 64)
    x = _conv_block(x, 128, dropout=0.2)        # → (B, 16,  32, 128)
    x = _conv_block(x, 256, dropout=0.3)        # → (B,  8,  16, 256)

    # Global Average Pooling → (B, 256)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name=name)
    return model


# Alias rétro-compatible : d'anciens scripts importaient ce nom.
CardioCNNLite = build_cnn_lite
