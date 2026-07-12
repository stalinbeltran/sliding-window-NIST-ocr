"""Dimensionador: CNN sobre una región (ventana) de la entrada 2D.

Ya no clasifica dígitos: describe el trazo que ve en la ventana con un vector de
descriptores geométricos interpretables (ver ``swnist.descriptors``). Ese mismo
vector es lo que consume el secuenciador, así que ``features(x)`` devuelve los
descriptores en su rango natural.
"""

import torch
import torch.nn as nn

from swnist.descriptors import NUM_DESCRIPTORS, activate


class Dimensionador(nn.Module):
    def __init__(
        self,
        window_size: int = 5,
        hidden_dim: int = 32,
        channels: tuple[int, ...] = (16, 32),
        feature_dim: int | None = None,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels or any(c <= 0 for c in channels):
            raise ValueError(
                f"model.channels debe ser una lista con al menos un entero positivo "
                f"(un canal por bloque convolucional); recibido: {list(channels)!r}.")
        # feature_dim queda fijo = nº de descriptores (lo que ve el secuenciador);
        # se guarda en la config para que `train_secuenciador` y la validación lo lean.
        self.config = {
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "channels": list(channels),
            "feature_dim": NUM_DESCRIPTORS,
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
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 9, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Cabeza de descriptores (salida cruda; la activación por canal la aplica
        # activate() al pasar a features/inferencia).
        self.head = nn.Linear(hidden_dim, NUM_DESCRIPTORS)

    @classmethod
    def from_config(cls, config: dict) -> "Dimensionador":
        return cls(
            window_size=config["window_size"],
            hidden_dim=config.get("hidden_dim", 32),
            channels=tuple(config["channels"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> descriptores crudos (B, 6) (para la pérdida)."""
        return self.head(self.trunk(self.conv(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> descriptores en rango natural (B, 6) (para el secuenciador)."""
        return activate(self.forward(x))

    # Alias legible: "describe la ventana".
    describe = features
