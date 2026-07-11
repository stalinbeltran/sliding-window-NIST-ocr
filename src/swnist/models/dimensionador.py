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
        channels: tuple[int, ...] = (16, 32),
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels or any(c <= 0 for c in channels):
            raise ValueError(
                f"model.channels debe ser una lista con al menos un entero positivo "
                f"(un canal por bloque convolucional); recibido: {list(channels)!r}.")
        self.config = {
            "window_size": window_size,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "channels": list(channels),
        }
        layers = []
        in_ch, size = 1, window_size
        for c in channels:
            layers += [nn.Conv2d(in_ch, c, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            # MaxPool solo mientras quede espacio (con muchas capas la ventana
            # llegaría a 0); el pooling adaptativo final fija la salida en 3×3.
            if size >= 2:
                layers.append(nn.MaxPool2d(2))
                size //= 2
            in_ch = c
        # Salida independiente del tamaño de ventana:
        layers.append(nn.AdaptiveAvgPool2d(3))
        self.conv = nn.Sequential(*layers)
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 9, feature_dim),
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
