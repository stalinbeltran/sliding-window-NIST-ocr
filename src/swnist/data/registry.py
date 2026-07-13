"""Catálogo de datasets y su compatibilidad con cada NN entrenable."""

from __future__ import annotations

from typing import Any

from swnist.data.datasets import MnistDigits

DATASETS: dict[str, dict[str, Any]] = {
    "mnist": {
        "label": "MNIST (imágenes 28x28, dígitos 0-9)",
        "cls": MnistDigits,
        "defaults": MnistDigits.defaults(),
        "nns": ["features_cnn"],
    },
}


def list_datasets(nn: str | None = None) -> list[dict[str, Any]]:
    """Datasets disponibles; si se pasa `nn`, solo los compatibles con esa NN."""
    out = []
    for name, info in DATASETS.items():
        if nn is not None and nn not in info["nns"]:
            continue
        out.append(
            {
                "name": name,
                "label": info["label"],
                "defaults": dict(info["defaults"]),
                "nns": list(info["nns"]),
            }
        )
    return out


def dataset_info(name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise KeyError(
            f"dataset '{name}' desconocido. Disponibles: {sorted(DATASETS)}"
        )
    return DATASETS[name]


def build_dataset(name: str, *, train: bool = True, **params: Any) -> Any:
    """Instancia un dataset validando que solo reciba SUS parámetros (regla 12)."""
    info = dataset_info(name)
    defaults = info["defaults"]
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ValueError(
            f"el dataset '{name}' no acepta el/los parámetro(s) {unknown}. "
            f"Acepta: {sorted(defaults)}."
        )
    merged = {**defaults, **params}
    return info["cls"](train=train, **merged)


def effective_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Params del dataset mezclados sobre sus defaults."""
    defaults = dict(dataset_info(name)["defaults"])
    defaults.update(params or {})
    return defaults
