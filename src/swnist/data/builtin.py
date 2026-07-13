"""Catálogo de datasets builtin (los que no son subconjuntos guardados)."""

from __future__ import annotations

from typing import Any

from swnist.data.datasets import MnistDigits

BUILTIN: dict[str, dict[str, Any]] = {
    "mnist": {
        "label": "MNIST (imágenes 28x28, dígitos 0-9)",
        "cls": MnistDigits,
        "defaults": MnistDigits.defaults(),
        "nns": ["features_cnn"],
    },
}


def build_builtin(name: str, *, train: bool, **params: Any) -> Any:
    info = BUILTIN[name]
    defaults = info["defaults"]
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ValueError(
            f"el dataset '{name}' no acepta el/los parámetro(s) {unknown}. "
            f"Acepta: {sorted(defaults)}."
        )
    return info["cls"](train=train, **{**defaults, **params})
