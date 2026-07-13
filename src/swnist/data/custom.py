"""Datasets custom: subconjuntos guardados de un dataset base.

Un dataset custom es una **definición versionable** (base + split + índices), no una
copia de las imágenes. Tiene dos orígenes:

- **Filtro de una evaluación**: las muestras que cumplen el filtro (aciertos / fallos /
  ambiguas, etiqueta real, predicción).
- **Recorte de un base**: las primeras `limit` muestras de un split.

Semántica al construirlo (la misma que espera el entrenamiento):
- `train=True`  → el subconjunto guardado (es el conjunto con el que se entrena/evalúa).
- `train=False` → el test completo del base (para el `final_test` del entrenamiento).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from swnist.data.builtin import BUILTIN, build_builtin

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CUSTOM_DIR = ROOT / "custom_datasets"

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _.-]{0,63}$")


class CustomSubset(Dataset):
    """Las muestras del subconjunto (o el test del base con `train=False`)."""

    def __init__(self, definition: dict[str, Any], *, train: bool = True):
        self.definition = definition
        base = definition["base"]
        if train:
            self.base = build_builtin(base, train=(definition["split"] == "train"))
            self.indices = list(definition["indices"])
        else:
            self.base = build_builtin(base, train=False)
            self.indices = list(range(len(self.base)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]

    def display_item(self, idx: int) -> np.ndarray:
        return self.base.display_item(self.indices[idx])

    def base_index(self, idx: int) -> int:
        """Índice de la muestra en el dataset base (para trazabilidad)."""
        return self.indices[idx]


class CustomDatasetStore:
    """CRUD sobre `custom_datasets/<nombre>.json`."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_CUSTOM_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.base_dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    # --- validación --------------------------------------------------------

    @staticmethod
    def validate_name(name: Any) -> str:
        if not isinstance(name, str) or not NAME_RE.match(name.strip()):
            raise ValueError(
                f"nombre de dataset inválido: {name!r}. Debe empezar por letra o número y "
                "usar solo letras, números, espacios, '_', '.' o '-' (máx. 64)."
            )
        return name.strip()

    # --- CRUD --------------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        base: str,
        split: str,
        indices: list[int],
        origin: dict[str, Any],
    ) -> dict[str, Any]:
        name = self.validate_name(name)
        if name in BUILTIN:
            raise ValueError(f"'{name}' es el nombre de un dataset builtin. Elige otro.")
        if self.exists(name):
            raise ValueError(f"ya existe un dataset custom llamado '{name}'.")
        if base not in BUILTIN:
            raise ValueError(f"dataset base '{base}' desconocido. Disponibles: {sorted(BUILTIN)}.")
        if split not in ("train", "test"):
            raise ValueError(f"split '{split}' inválido: debe ser 'train' o 'test'.")
        if not indices:
            raise ValueError(
                "el subconjunto quedaría vacío: no hay ninguna muestra que cumpla el filtro."
            )

        definition = {
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base": base,
            "split": split,
            "indices": sorted(int(i) for i in indices),
            "size": len(indices),
            "origin": origin,
        }
        self._write(name, definition)
        return definition

    def get(self, name: str) -> dict[str, Any]:
        path = self.path(name)
        if not path.exists():
            raise KeyError(f"dataset custom '{name}' no encontrado")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list(self) -> list[dict[str, Any]]:
        """Lo más reciente primero."""
        items = [self.get(path.stem) for path in self.base_dir.glob("*.json")]
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def rename(self, name: str, new_name: str) -> dict[str, Any]:
        definition = self.get(name)
        new_name = self.validate_name(new_name)
        if new_name == name:
            return definition
        if new_name in BUILTIN or self.exists(new_name):
            raise ValueError(f"ya existe un dataset llamado '{new_name}'.")
        definition["name"] = new_name
        self._write(new_name, definition)
        self.path(name).unlink()
        return definition

    def delete(self, name: str) -> None:
        if not self.exists(name):
            raise KeyError(f"dataset custom '{name}' no encontrado")
        self.path(name).unlink()

    def build(self, name: str, *, train: bool = True) -> CustomSubset:
        return CustomSubset(self.get(name), train=train)

    # --- helpers -----------------------------------------------------------

    def _write(self, name: str, definition: dict[str, Any]) -> None:
        with self.path(name).open("w", encoding="utf-8") as handle:
            json.dump(definition, handle, indent=2, ensure_ascii=False)
