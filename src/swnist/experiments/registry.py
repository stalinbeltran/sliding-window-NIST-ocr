"""Registro de experimentos: crear, listar, leer, métricas y CRUD (alias/copiar/borrar).

Cada experimento vive en `experiments/<id>/` con:
  config.json, environment.json, status.json, metrics.jsonl, checkpoints/{best,last}.pt
El id tiene la forma `exp_YYYYMMDD_HHMMSS_<nn>` (el timestamp lo hace ordenable).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from swnist.repro import capture_environment

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_DIR = ROOT / "experiments"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentRegistry:
    """Acceso programático al registro (`python -m swnist.experiments.registry`)."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_EXPERIMENTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --- rutas -------------------------------------------------------------

    def path(self, exp_id: str) -> Path:
        return self.base_dir / exp_id

    def checkpoint_path(self, exp_id: str, name: str = "best.pt") -> Path:
        return self.path(exp_id) / "checkpoints" / name

    # --- creación ----------------------------------------------------------

    def new_id(self, nn: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{stamp}_{nn}"
        suffix = 1
        while self.path(exp_id).exists():
            suffix += 1
            exp_id = f"exp_{stamp}_{nn}_{suffix}"
        return exp_id

    def create(self, config: dict[str, Any], *, label: str | None = None) -> str:
        exp_id = self.new_id(config["nn"])
        directory = self.path(exp_id)
        (directory / "checkpoints").mkdir(parents=True)
        self._write_json(directory / "config.json", config)
        self._write_json(directory / "environment.json", capture_environment())
        self._write_json(
            directory / "status.json",
            {
                "id": exp_id,
                "label": label or exp_id,
                "nn": config["nn"],
                "status": "created",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "best": None,
                "final_test": None,
                "error": None,
            },
        )
        (directory / "metrics.jsonl").touch()
        return exp_id

    # --- lectura -----------------------------------------------------------

    def exists(self, exp_id: str) -> bool:
        return (self.path(exp_id) / "status.json").exists()

    def get(self, exp_id: str) -> dict[str, Any]:
        directory = self.path(exp_id)
        if not (directory / "status.json").exists():
            raise KeyError(f"experimento '{exp_id}' no encontrado")
        status = self._read_json(directory / "status.json")
        return {
            **status,
            "config": self._read_json(directory / "config.json"),
            "environment": self._read_json(directory / "environment.json"),
            "has_checkpoint": self.checkpoint_path(exp_id).exists(),
        }

    def list(self) -> list[dict[str, Any]]:
        """Lo más reciente primero (el id lleva timestamp → orden lexicográfico inverso)."""
        items = []
        for directory in self.base_dir.iterdir():
            if (directory / "status.json").exists():
                status = self._read_json(directory / "status.json")
                status["has_checkpoint"] = self.checkpoint_path(directory.name).exists()
                status["config"] = self._read_json(directory / "config.json")
                items.append(status)
        return sorted(items, key=lambda item: item["id"], reverse=True)

    def metrics(self, exp_id: str, *, since: int = 0) -> list[dict[str, Any]]:
        path = self.path(exp_id) / "metrics.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in list(handle)[since:] if line.strip()]

    # --- escritura durante el entrenamiento --------------------------------

    def update_status(self, exp_id: str, **fields: Any) -> dict[str, Any]:
        path = self.path(exp_id) / "status.json"
        status = self._read_json(path)
        status.update(fields)
        self._write_json(path, status)
        return status

    def log_metric(self, exp_id: str, record: dict[str, Any]) -> None:
        path = self.path(exp_id) / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # --- CRUD --------------------------------------------------------------

    def rename(self, exp_id: str, label: str) -> dict[str, Any]:
        """Renombrar es solo un alias: el id nunca cambia (otros lo referencian)."""
        if not self.exists(exp_id):
            raise KeyError(f"experimento '{exp_id}' no encontrado")
        label = label.strip()
        if not label:
            raise ValueError("el nombre no puede estar vacío")
        return self.update_status(exp_id, label=label)

    def duplicate(self, exp_id: str) -> str:
        source = self.path(exp_id)
        if not (source / "status.json").exists():
            raise KeyError(f"experimento '{exp_id}' no encontrado")
        config = self._read_json(source / "config.json")
        new_id = self.new_id(config["nn"])
        shutil.copytree(source, self.path(new_id))
        status = self._read_json(self.path(new_id) / "status.json")
        status["id"] = new_id
        status["label"] = f"{status.get('label', exp_id)} (copia)"
        status["created_at"] = _now()
        self._write_json(self.path(new_id) / "status.json", status)
        return new_id

    def delete(self, exp_id: str) -> None:
        if not self.exists(exp_id):
            raise KeyError(f"experimento '{exp_id}' no encontrado")
        status = self.get(exp_id)
        if status["status"] == "running":
            raise ValueError(
                f"el experimento '{exp_id}' está corriendo. Deténlo antes de eliminarlo."
            )
        users = [
            other["id"]
            for other in self.list()
            if other["id"] != exp_id and other["config"].get("init_from") == exp_id
        ]
        if users:
            raise ValueError(
                f"no se puede eliminar '{exp_id}': lo usan como init_from "
                f"{users}. Elimina primero esos experimentos."
            )
        shutil.rmtree(self.path(exp_id))

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_experiments(base_dir: Path | str | None = None) -> Iterator[dict[str, Any]]:
    yield from ExperimentRegistry(base_dir).list()


def _main() -> None:
    # La consola de Windows es cp1252: sin esto, una etiqueta con acentos o flechas
    # rompe el listado.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    registry = ExperimentRegistry()
    for exp in registry.list():
        best = exp.get("best") or {}
        acc = best.get("val_acc")
        acc_txt = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        print(f"{exp['id']:<40} {exp['status']:<10} val_acc={acc_txt:<8} {exp.get('label', '')}")


if __name__ == "__main__":
    _main()
