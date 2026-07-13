"""Catálogo de datasets (builtin + custom) y su compatibilidad con cada NN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swnist.data.builtin import BUILTIN, build_builtin
from swnist.data.custom import CustomDatasetStore

_store: CustomDatasetStore | None = None


def store() -> CustomDatasetStore:
    """Store de datasets custom (singleton; `configure()` lo redirige en los tests)."""
    global _store
    if _store is None:
        _store = CustomDatasetStore()
    return _store


def configure(custom_dir: Path | str | None) -> CustomDatasetStore:
    """Apunta el store a otro directorio (la web app y los tests lo usan)."""
    global _store
    _store = CustomDatasetStore(custom_dir)
    return _store


def is_custom(name: str) -> bool:
    return name not in BUILTIN and store().exists(name)


def _custom_entry(definition: dict[str, Any]) -> dict[str, Any]:
    name = definition["name"]
    base = definition["base"]
    origin = definition.get("origin", {})
    return {
        "name": name,
        "label": (
            f"{name} — custom de {base}/{definition['split']}, "
            f"{definition['size']} muestras"
        ),
        # Un custom ya tiene sus muestras fijadas por índice: no admite params.
        "defaults": {},
        "nns": list(BUILTIN[base]["nns"]),
        "is_custom": True,
        "base": base,
        "split": definition["split"],
        "size": definition["size"],
        "created_at": definition["created_at"],
        "origin": origin,
    }


def list_datasets(nn: str | None = None) -> list[dict[str, Any]]:
    """Datasets disponibles (builtin primero, customs del más nuevo al más viejo).
    Si se pasa `nn`, solo los compatibles con esa NN."""
    out: list[dict[str, Any]] = []
    for name, info in BUILTIN.items():
        if nn is None or nn in info["nns"]:
            out.append(
                {
                    "name": name,
                    "label": info["label"],
                    "defaults": dict(info["defaults"]),
                    "nns": list(info["nns"]),
                    "is_custom": False,
                }
            )
    for definition in store().list():
        entry = _custom_entry(definition)
        if nn is None or nn in entry["nns"]:
            out.append(entry)
    return out


def dataset_info(name: str) -> dict[str, Any]:
    if name in BUILTIN:
        return {**BUILTIN[name], "is_custom": False}
    if store().exists(name):
        return {**_custom_entry(store().get(name)), "cls": None}
    known = sorted(BUILTIN) + [d["name"] for d in store().list()]
    raise KeyError(f"dataset '{name}' desconocido. Disponibles: {known}")


def build_dataset(name: str, *, train: bool = True, **params: Any) -> Any:
    """Instancia un dataset validando que solo reciba SUS parámetros (regla 12)."""
    if name in BUILTIN:
        return build_builtin(name, train=train, **params)
    if store().exists(name):
        if params:
            raise ValueError(
                f"el dataset custom '{name}' no acepta parámetros {sorted(params)}: sus "
                "muestras ya están fijadas por índice. Crea otro dataset si necesitas otras."
            )
        return store().build(name, train=train)
    dataset_info(name)  # lanza KeyError con la lista de disponibles


def effective_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Params del dataset mezclados sobre sus defaults."""
    defaults = dict(dataset_info(name)["defaults"])
    defaults.update(params or {})
    return defaults


def compatible_nns(name: str) -> list[str]:
    return list(dataset_info(name)["nns"])
