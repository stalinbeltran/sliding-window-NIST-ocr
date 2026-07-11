"""Registro de experimentos: creación, estado, métricas y listado programático.

Cada experimento vive en experiments/<id>/ con:
  config.json, environment.json, status.json, metrics.jsonl, checkpoints/ (git-ignored)

Uso programático:
    from swnist.experiments.registry import ExperimentRegistry
    reg = ExperimentRegistry()
    reg.list_experiments()

CLI:
    python -m swnist.experiments.registry
"""

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from swnist import EXPERIMENTS_DIR
from swnist.repro import collect_environment

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else EXPERIMENTS_DIR
        self.root.mkdir(exist_ok=True)

    # ---------- creación ----------

    def create_experiment(self, nn: str, config: dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{ts}_{nn}"
        # Evitar colisión si se crean dos en el mismo segundo
        suffix = 1
        while (self.root / exp_id).exists():
            exp_id = f"exp_{ts}_{nn}_{suffix}"
            suffix += 1
        d = self.root / exp_id
        (d / "checkpoints").mkdir(parents=True)
        self._write_json(d / "config.json", {"nn": nn, **config})
        self._write_json(d / "environment.json", collect_environment())
        self._write_json(d / "status.json", {
            "status": "created",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "best": None,
            "final_test": None,
            "error": None,
        })
        (d / "metrics.jsonl").touch()
        return exp_id

    # ---------- escritura ----------

    def log_metrics(self, exp_id: str, record: dict) -> None:
        with _lock:
            with open(self.root / exp_id / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def update_status(self, exp_id: str, **fields) -> dict:
        with _lock:
            path = self.root / exp_id / "status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            status.update(fields)
            self._write_json(path, status)
            return status

    def mark_running(self, exp_id: str) -> None:
        self.update_status(exp_id, status="running", started_at=_now())

    def mark_finished(self, exp_id: str, status: str = "completed", **fields) -> None:
        self.update_status(exp_id, status=status, finished_at=_now(), **fields)

    def checkpoints_dir(self, exp_id: str) -> Path:
        return self.root / exp_id / "checkpoints"

    # ---------- CRUD ----------

    def set_label(self, exp_id: str, label: str) -> dict:
        """Alias visible del experimento. El id no se renombra porque otros
        experimentos (secuenciador→dimensionador) lo referencian por id."""
        self.get_experiment(exp_id)  # FileNotFoundError si no existe
        return self.update_status(exp_id, label=label)

    def referencing_experiments(self, exp_id: str) -> list[str]:
        """Experimentos cuya config referencia a `exp_id` como dimensionador."""
        return [e["id"] for e in self.list_experiments()
                if e["config"].get("dimensionador_experiment") == exp_id]

    def delete_experiment(self, exp_id: str) -> None:
        d = self.root / exp_id
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"Experimento no encontrado: {exp_id!r}")
        shutil.rmtree(d)

    def copy_experiment(self, exp_id: str) -> str:
        """Duplica un experimento (config, métricas y checkpoints) con id nuevo."""
        src = self.root / exp_id
        if not (src / "config.json").exists():
            raise FileNotFoundError(f"Experimento no encontrado: {exp_id!r}")
        nn = json.loads((src / "config.json").read_text(encoding="utf-8")).get("nn", "nn")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_id = f"exp_{ts}_{nn}"
        suffix = 1
        while (self.root / new_id).exists():
            new_id = f"exp_{ts}_{nn}_{suffix}"
            suffix += 1
        shutil.copytree(src, self.root / new_id)
        self.update_status(new_id, label=f"copia de {exp_id}")
        return new_id

    # ---------- lectura ----------

    def list_experiments(self, nn: str | None = None, status: str | None = None) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not (d / "config.json").exists():
                continue
            info = self.get_experiment(d.name)
            if nn and info["config"].get("nn") != nn:
                continue
            if status and info["status"].get("status") != status:
                continue
            out.append(info)
        return out

    def get_experiment(self, exp_id: str) -> dict:
        d = self.root / exp_id
        return {
            "id": exp_id,
            "config": json.loads((d / "config.json").read_text(encoding="utf-8")),
            "status": json.loads((d / "status.json").read_text(encoding="utf-8")),
            "environment": json.loads((d / "environment.json").read_text(encoding="utf-8")),
        }

    def get_metrics(self, exp_id: str) -> list[dict]:
        path = self.root / exp_id / "metrics.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    reg = ExperimentRegistry()
    exps = reg.list_experiments()
    if not exps:
        print("No hay experimentos registrados.")
    for e in exps:
        s = e["status"]
        best = s.get("best") or {}
        print(f"{e['id']:45s} {s['status']:10s} "
              f"val_acc={best.get('val_acc', '—')} test={s.get('final_test') or '—'}")
