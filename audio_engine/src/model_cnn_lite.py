
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc Conv2d → BatchNorm → ReLU → MaxPool (+ Dropout2d optionnel)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 pool: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CardioCNNLite(nn.Module):

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  kernel=3, pool=2, dropout=0.0),   # → (B, 32,  64, 128)
            ConvBlock(32,  64,  kernel=3, pool=2, dropout=0.1),   # → (B, 64,  32,  64)
            ConvBlock(64,  128, kernel=3, pool=2, dropout=0.2),   # → (B, 128, 16,  32)
            ConvBlock(128, 256, kernel=3, pool=2, dropout=0.3),   # → (B, 256,  8,  16)
        )
        # Global Average Pooling → (B, 256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)