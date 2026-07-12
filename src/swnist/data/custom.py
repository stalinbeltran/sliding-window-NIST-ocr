"""Datasets custom: subconjuntos filtrados de un dataset base, con CRUD.

Un dataset custom es un JSON en `custom_datasets/<nombre>.json` (versionado) con:
  - base: {name, params, split, seed} → dataset builtin del que se tomaron muestras
  - indices: índices dentro de ese dataset base (reproducibles dado params+seed)
  - description, created_at, source (p. ej. evaluación + filtro que lo originó)

Semántica al construir (`build_dataset`):
  - train=True  → subconjunto (base con su split/seed guardados, solo `indices`)
  - train=False → split de test completo del dataset base (para el test final)
Los params pasados se fusionan sobre los del base (p. ej. el secuenciador impone
`window_size`); los índices refieren a imágenes, así que siguen siendo válidos.
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import Dataset

from swnist import CUSTOM_DATASETS_DIR

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def slugify(name: str) -> str:
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9_-]+", "_", norm.strip().lower()).strip("_")
    return slug


class CustomSubset(Dataset):
    """Subconjunto de un dataset base según una definición custom."""

    def __init__(self, base_dataset, indices: list[int]):
        self.base = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def display_item(self, idx):
        return self.base.display_item(self.indices[idx])

    def trajectory(self, idx):
        """Recorrido de la ventana, delegado al base (solo lo tienen los
        datasets de secuencias)."""
        return self.base.trajectory(self.indices[idx])


class CustomDatasetStore:
    """CRUD de datasets custom sobre `custom_datasets/` (root parametrizable en tests)."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CUSTOM_DATASETS_DIR
        self.root.mkdir(exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(self._summary(d))
        return out

    def get(self, name: str) -> dict:
        p = self._path(name)
        if not p.exists():
            raise KeyError(f"Dataset custom no encontrado: {name!r}")
        return json.loads(p.read_text(encoding="utf-8"))

    def get_summary(self, name: str) -> dict:
        return self._summary(self.get(name))

    def create(self, base: dict, indices: list[int], name: str | None = None,
               description: str = "", source: dict | None = None) -> dict:
        if not indices:
            raise ValueError("El filtro no produjo ninguna muestra: no se crea un "
                             "dataset vacío (relaja el filtro).")
        if name:
            name = slugify(name)
            if not name or not _NAME_RE.match(name):
                raise ValueError(
                    "Nombre inválido: usa minúsculas, dígitos, '_' o '-' (sin espacios).")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = slugify(f"{base['name']}_{ts}")
        if self.exists(name):
            raise ValueError(f"Ya existe un dataset custom llamado {name!r}: elige otro nombre.")
        definition = {
            "name": name,
            "description": description,
            "base": base,  # {name, params, split, seed}
            "indices": [int(i) for i in indices],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source or {},
        }
        self._write(name, definition)
        return self._summary(definition)

    def rename(self, name: str, new_name: str) -> dict:
        d = self.get(name)
        new_name = slugify(new_name)
        if not new_name or not _NAME_RE.match(new_name):
            raise ValueError("Nombre inválido: usa minúsculas, dígitos, '_' o '-'.")
        if new_name == name:
            return self._summary(d)
        if self.exists(new_name):
            raise ValueError(f"Ya existe un dataset custom llamado {new_name!r}.")
        d["name"] = new_name
        self._write(new_name, d)
        self._path(name).unlink()
        return self._summary(d)

    def update_description(self, name: str, description: str) -> dict:
        d = self.get(name)
        d["description"] = description
        self._write(name, d)
        return self._summary(d)

    def copy(self, name: str, new_name: str | None = None) -> dict:
        d = self.get(name)
        if new_name:
            new_name = slugify(new_name)
            if self.exists(new_name):
                raise ValueError(f"Ya existe un dataset custom llamado {new_name!r}.")
        else:
            new_name, n = f"{name}_copia", 2
            while self.exists(new_name):
                new_name = f"{name}_copia{n}"
                n += 1
        d = dict(d, name=new_name,
                 created_at=datetime.now(timezone.utc).isoformat(),
                 source={**d.get("source", {}), "copied_from": name})
        self._write(new_name, d)
        return self._summary(d)

    def delete(self, name: str) -> None:
        p = self._path(name)
        if not p.exists():
            raise KeyError(f"Dataset custom no encontrado: {name!r}")
        p.unlink()

    def _write(self, name: str, definition: dict) -> None:
        self._path(name).write_text(
            json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _summary(d: dict) -> dict:
        return {
            "name": d["name"],
            "description": d.get("description", ""),
            "base": d["base"],
            "count": len(d["indices"]),
            "created_at": d.get("created_at"),
            "source": d.get("source", {}),
        }
