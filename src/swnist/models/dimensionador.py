"""Dimensionador: CNN sobre una región (ventana) de la entrada 2D.

Produce un vector de características (`features`) que consume el secuenciador,
y logits de clasificación de dígitos usados para pre-entrenarlo.
"""

import torch
import torch.nn as nn


class Dimensionador(nn.Module):
    def __init__(
        self,
        window_size: int = 14,
        feature_dim: int = 32,
        num_classes: int = 10,
        channels: tuple[int, int] = (16, 32),
    ):
        super().__init__()
        c1, c2 = channels
        self.config = {
            "window_size": window_size,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "channels": [c1, c2],
        }
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Salida independiente del tamaño de ventana:
            nn.AdaptiveAvgPool2d(3),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 9, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    @classmethod
    def from_config(cls, config: dict) -> "Dimensionador":
        return cls(
            window_size=config["window_size"],
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"],
            channels=tuple(config["channels"]),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> (B, feature_dim)."""
        return self.feature_head(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> logits (B, num_classes)."""
        return self.classifier(self.features(x))
