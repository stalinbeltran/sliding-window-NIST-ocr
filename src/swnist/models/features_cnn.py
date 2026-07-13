"""CNN configurable por JSON: N capas conv (kernels y tamaño de kernel por capa),
pooling opcional por capa y capa de salida.

Además de clasificar, expone los **feature maps** de cada capa (las activaciones que
la red "reconoce" en la imagen) y los **kernels aprendidos**, para representarlos
como matrices en la web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "none": nn.Identity,
}

POOL_TYPES = ("max", "avg")


@dataclass(frozen=True)
class LayerSpec:
    """Una capa convolucional (+ activación + pooling opcional)."""

    kernels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: str = "relu"
    pool: dict[str, Any] | None = None  # {"type": "max"|"avg", "size": int, "stride": int|None}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LayerSpec":
        return cls(
            kernels=raw["kernels"],
            kernel_size=raw["kernel_size"],
            stride=raw.get("stride", 1),
            padding=raw.get("padding", 0),
            activation=raw.get("activation", "relu"),
            pool=raw.get("pool"),
        )


def conv_out_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


def pool_out_size(size: int, pool: dict[str, Any]) -> int:
    pool_size = pool["size"]
    stride = pool.get("stride") or pool_size
    return (size - pool_size) // stride + 1


def spatial_trace(layers: Iterable[LayerSpec], input_size: int) -> list[dict[str, int]]:
    """Tamaño espacial tras cada capa. Lanza ValueError si el mapa se agota."""
    size = input_size
    trace: list[dict[str, int]] = []
    for i, spec in enumerate(layers):
        after_conv = conv_out_size(size, spec.kernel_size, spec.stride, spec.padding)
        if after_conv < 1:
            raise ValueError(
                f"la capa {i + 1} (kernel {spec.kernel_size}x{spec.kernel_size}, "
                f"stride {spec.stride}, padding {spec.padding}) no cabe en un mapa de "
                f"{size}x{size}: el resultado sería {after_conv}x{after_conv}. "
                "Reduce el tamaño del kernel, sube el padding o quita capas/pooling previos."
            )
        after_pool = after_conv
        if spec.pool is not None:
            after_pool = pool_out_size(after_conv, spec.pool)
            if after_pool < 1:
                pool_size = spec.pool["size"]
                raise ValueError(
                    f"el pooling {pool_size}x{pool_size} de la capa {i + 1} no cabe en un "
                    f"mapa de {after_conv}x{after_conv}. Quita el pooling de esa capa o "
                    "reduce su tamaño."
                )
        trace.append({"conv": after_conv, "out": after_pool})
        size = after_pool
    return trace


class FeaturesCNN(nn.Module):
    """CNN de N capas configurables + capa de salida (clasificación de dígitos).

    - `forward(x)` → logits (N, num_classes).
    - `feature_maps(x)` → activaciones por capa, que son los *features reconocidos*.
    - `kernels()` → pesos de cada conv, para verlos como matrices.
    """

    def __init__(
        self,
        layers: list[LayerSpec],
        *,
        input_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("la CNN necesita al menos 1 capa convolucional")
        self.layer_specs = list(layers)
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        trace = spatial_trace(self.layer_specs, input_size)

        blocks: list[nn.Module] = []
        channels = in_channels
        for spec in self.layer_specs:
            ops: list[nn.Module] = [
                nn.Conv2d(
                    channels,
                    spec.kernels,
                    kernel_size=spec.kernel_size,
                    stride=spec.stride,
                    padding=spec.padding,
                ),
                ACTIVATIONS[spec.activation](),
            ]
            if spec.pool is not None:
                pool_size = spec.pool["size"]
                pool_stride = spec.pool.get("stride") or pool_size
                cls = nn.MaxPool2d if spec.pool["type"] == "max" else nn.AvgPool2d
                ops.append(cls(kernel_size=pool_size, stride=pool_stride))
            blocks.append(nn.Sequential(*ops))
            channels = spec.kernels
        self.blocks = nn.ModuleList(blocks)

        final_size = trace[-1]["out"]
        self.feature_dim = channels * final_size * final_size
        head: list[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        if hidden_dim > 0:
            head.append(nn.Linear(self.feature_dim, hidden_dim))
            head.append(nn.ReLU())
            head.append(nn.Linear(hidden_dim, num_classes))
        else:
            head.append(nn.Linear(self.feature_dim, num_classes))
        self.head = nn.Sequential(*head)

    # --- API ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Salida de cada capa (activación + pooling): (N, kernels, H, W) por capa.

        Son los *features reconocidos*: cada canal es la respuesta de un kernel sobre
        la imagen, y se representa como una matriz H×W.
        """
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            maps.append(x)
        return maps

    def kernels(self) -> list[torch.Tensor]:
        """Pesos aprendidos de cada conv: (kernels, in_channels, k, k) por capa."""
        return [block[0].weight.detach() for block in self.blocks]

    def layer_shapes(self) -> list[dict[str, int]]:
        trace = spatial_trace(self.layer_specs, self.input_size)
        return [
            {
                "layer": i + 1,
                "kernels": spec.kernels,
                "kernel_size": spec.kernel_size,
                "conv_size": t["conv"],
                "out_size": t["out"],
                "pooled": spec.pool is not None,
            }
            for i, (spec, t) in enumerate(zip(self.layer_specs, trace))
        ]

    # --- construcción desde config ----------------------------------------

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> "FeaturesCNN":
        layers = [LayerSpec.from_dict(raw) for raw in model_cfg["layers"]]
        return cls(
            layers,
            input_size=model_cfg.get("input_size", 28),
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 10),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.0),
        )
