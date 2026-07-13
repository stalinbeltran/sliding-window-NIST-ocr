"""Catálogo de redes entrenables y su configuración por defecto."""

from __future__ import annotations

import copy
from typing import Any

NNS: dict[str, dict[str, Any]] = {
    "features_cnn": {
        "label": "CNN de features (clasifica dígitos y expone sus feature maps)",
        "description": (
            "CNN configurable: N capas convolucionales (kernels y tamaño de kernel por "
            "capa), pooling opcional por capa y capa de salida de 10 clases. Una vez "
            "entrenada, se extraen sus feature maps para verlos como matrices."
        ),
        "default_config": {
            "nn": "features_cnn",
            "dataset": {"name": "mnist", "params": {"limit": None}},
            "model": {
                "input_size": 28,
                "in_channels": 1,
                "num_classes": 10,
                "hidden_dim": 64,
                "dropout": 0.0,
                "layers": [
                    {
                        "kernels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "activation": "relu",
                        "pool": {"type": "max", "size": 2},
                    },
                    {
                        "kernels": 16,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "activation": "relu",
                        "pool": {"type": "max", "size": 2},
                    },
                ],
            },
            "training": {
                "epochs": 3,
                "batch_size": 128,
                "lr": 0.001,
                "val_fraction": 0.1,
                "log_every": 50,
            },
            "seed": 1234,
            "init_from": None,
        },
    },
}


def list_nns() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": info["label"],
            "description": info["description"],
            "default_config": copy.deepcopy(info["default_config"]),
        }
        for name, info in NNS.items()
    ]


def default_config(nn: str) -> dict[str, Any]:
    if nn not in NNS:
        raise KeyError(f"NN '{nn}' desconocida. Disponibles: {sorted(NNS)}")
    return copy.deepcopy(NNS[nn]["default_config"])
