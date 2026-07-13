"""Registro de evaluaciones: `evaluations/<eval_id>/` con config, estado y resultados.

- `config.json`  — qué NN, sobre qué dataset/split y con qué umbral de ambigüedad.
- `status.json`  — estado, resumen (accuracy, fallos, ambiguas) y matriz de confusión.
- `results.jsonl`— una línea por muestra (label, pred, conf, margin) — git-ignored.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALUATIONS_DIR = ROOT / "evaluations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def matches(result: dict[str, Any], filtro: dict[str, Any]) -> bool:
    """¿La muestra cumple el filtro? (resultado + clase real + clase predicha)"""
    outcome = filtro.get("outcome", "all")
    if outcome == "correct" and not result["correct"]:
        return False
    if outcome == "failed" and result["correct"]:
        return False
    if outcome == "ambiguous" and not result["ambiguous"]:
        return False
    if filtro.get("label") is not None and result["label"] != filtro["label"]:
        return False
    if filtro.get("pred") is not None and result["pred"] != filtro["pred"]:
        return False
    return True


class EvaluationRegistry:
    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_EVALUATIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --- rutas -------------------------------------------------------------

    def path(self, eval_id: str) -> Path:
        return self.base_dir / eval_id

    def results_path(self, eval_id: str) -> Path:
        return self.path(eval_id) / "results.jsonl"

    # --- creación ----------------------------------------------------------

    def new_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{stamp}"
        suffix = 1
        while self.path(eval_id).exists():
            suffix += 1
            eval_id = f"eval_{stamp}_{suffix}"
        return eval_id

    def create(self, config: dict[str, Any], *, label: str | None = None) -> str:
        eval_id = self.new_id()
        directory = self.path(eval_id)
        directory.mkdir(parents=True)
        self._write_json(directory / "config.json", config)
        self._write_json(
            directory / "status.json",
            {
                "id": eval_id,
                "label": label or eval_id,
                "experiment": config["experiment"],
                "dataset": config["dataset"],
                "split": config["split"],
                "status": "created",
                "created_at": _now(),
                "finished_at": None,
                "summary": None,
                "confusion": None,
                "error": None,
            },
        )
        self.results_path(eval_id).touch()
        return eval_id

    # --- lectura -----------------------------------------------------------

    def exists(self, eval_id: str) -> bool:
        return (self.path(eval_id) / "status.json").exists()

    def get(self, eval_id: str) -> dict[str, Any]:
        directory = self.path(eval_id)
        if not (directory / "status.json").exists():
            raise KeyError(f"evaluación '{eval_id}' no encontrada")
        return {
            **self._read_json(directory / "status.json"),
            "config": self._read_json(directory / "config.json"),
        }

    def list(self, *, experiment: str | None = None) -> list[dict[str, Any]]:
        """Lo más reciente primero (el id lleva timestamp)."""
        items = []
        for directory in self.base_dir.iterdir():
            if not (directory / "status.json").exists():
                continue
            status = self._read_json(directory / "status.json")
            if experiment is not None and status["experiment"] != experiment:
                continue
            items.append(status)
        return sorted(items, key=lambda item: item["id"], reverse=True)

    def iter_results(self, eval_id: str) -> Iterator[dict[str, Any]]:
        path = self.results_path(eval_id)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def results(
        self,
        eval_id: str,
        *,
        filtro: dict[str, Any] | None = None,
        offset: int = 0,
        limit: int = 48,
    ) -> dict[str, Any]:
        """Resultados que cumplen el filtro, paginados. `total` es el nº de coincidencias."""
        filtro = filtro or {"outcome": "all"}
        matched: list[dict[str, Any]] = []
        for result in self.iter_results(eval_id):
            if matches(result, filtro):
                matched.append(result)
        page = matched[offset : offset + limit]
        return {
            "id": eval_id,
            "filter": filtro,
            "total": len(matched),
            "offset": offset,
            "limit": limit,
            "results": page,
        }

    def matching_indices(self, eval_id: str, filtro: dict[str, Any]) -> list[int]:
        """Índices en el dataset BASE de las muestras que cumplen el filtro.

        Es lo que necesita un dataset custom para materializar el filtro.
        """
        return [
            result["base_index"]
            for result in self.iter_results(eval_id)
            if matches(result, filtro)
        ]

    # --- escritura ---------------------------------------------------------

    def update_status(self, eval_id: str, **fields: Any) -> dict[str, Any]:
        path = self.path(eval_id) / "status.json"
        status = self._read_json(path)
        status.update(fields)
        self._write_json(path, status)
        return status

    def append_results(self, eval_id: str, records: list[dict[str, Any]]) -> None:
        with self.results_path(eval_id).open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    # --- CRUD --------------------------------------------------------------

    def rename(self, eval_id: str, label: str) -> dict[str, Any]:
        if not self.exists(eval_id):
            raise KeyError(f"evaluación '{eval_id}' no encontrada")
        label = label.strip()
        if not label:
            raise ValueError("el nombre no puede estar vacío")
        return self.update_status(eval_id, label=label)

    def delete(self, eval_id: str) -> None:
        if not self.exists(eval_id):
            raise KeyError(f"evaluación '{eval_id}' no encontrada")
        if self.get(eval_id)["status"] == "running":
            raise ValueError(
                f"la evaluación '{eval_id}' está corriendo. Espera a que termine."
            )
        shutil.rmtree(self.path(eval_id))

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
