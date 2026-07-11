"""Registro de evaluaciones: cada una vive en evaluations/<id>/ con
config.json, status.json y results.jsonl (una línea por muestra evaluada).

Resultado por muestra:
  {"index", "label", "pred", "correct", "conf", "margin"} (+ "pred_steps" en el
  secuenciador). `conf` = probabilidad de la clase predicha; `margin` = p1 - p2
  (diferencia entre las dos clases más probables): un margen bajo = ambigüedad.
"""

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from swnist import EVALUATIONS_DIR

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else EVALUATIONS_DIR
        self.root.mkdir(exist_ok=True)

    def create_evaluation(self, config: dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{ts}"
        suffix = 1
        while (self.root / eval_id).exists():
            eval_id = f"eval_{ts}_{suffix}"
            suffix += 1
        d = self.root / eval_id
        d.mkdir(parents=True)
        (d / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_status(eval_id, {
            "status": "created", "created_at": _now(), "started_at": None,
            "finished_at": None, "progress": None, "summary": None, "error": None,
        })
        (d / "results.jsonl").touch()
        return eval_id

    def update_status(self, eval_id: str, **fields) -> dict:
        with _lock:
            path = self.root / eval_id / "status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            status.update(fields)
            path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            return status

    def _write_status(self, eval_id: str, status: dict) -> None:
        (self.root / eval_id / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_results(self, eval_id: str, records: list[dict]) -> None:
        with _lock:
            with open(self.root / eval_id / "results.jsonl", "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    def list_evaluations(self, experiment: str | None = None) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not (d / "config.json").exists():
                continue
            info = self.get_evaluation(d.name)
            if experiment and info["config"].get("experiment") != experiment:
                continue
            out.append(info)
        return out

    def get_evaluation(self, eval_id: str) -> dict:
        d = self.root / eval_id
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"Evaluación no encontrada: {eval_id!r}")
        return {
            "id": eval_id,
            "config": json.loads((d / "config.json").read_text(encoding="utf-8")),
            "status": json.loads((d / "status.json").read_text(encoding="utf-8")),
        }

    def get_results(self, eval_id: str) -> list[dict]:
        path = self.root / eval_id / "results.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    def filter_results(self, eval_id: str, result: str = "all",
                       label: int | None = None, pred: int | None = None,
                       ambiguity: float = 0.2) -> list[dict]:
        """Filtra resultados: result ∈ all|correct|failed|ambiguous.

        `ambiguous` = margen (p1 - p2) menor que el umbral `ambiguity`, es decir,
        la red dudó entre al menos dos clases (independiente de si acertó).
        """
        rows = self.get_results(eval_id)
        if result == "correct":
            rows = [r for r in rows if r["correct"]]
        elif result == "failed":
            rows = [r for r in rows if not r["correct"]]
        elif result == "ambiguous":
            rows = [r for r in rows if r["margin"] < ambiguity]
        elif result != "all":
            raise ValueError(f"Filtro desconocido: {result!r} "
                             f"(usa all|correct|failed|ambiguous).")
        if label is not None:
            rows = [r for r in rows if r["label"] == label]
        if pred is not None:
            rows = [r for r in rows if r["pred"] == pred]
        return rows

    def delete(self, eval_id: str) -> None:
        d = self.root / eval_id
        if not d.exists():
            raise FileNotFoundError(f"Evaluación no encontrada: {eval_id!r}")
        shutil.rmtree(d)
