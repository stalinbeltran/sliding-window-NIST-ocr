"""Jobs en hilos (entrenamientos y evaluaciones), con parada cooperativa."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from swnist.evaluation.registry import EvaluationRegistry
from swnist.evaluation.runner import run_evaluation
from swnist.experiments.registry import ExperimentRegistry
from swnist.training.train_features_cnn import run_training
from swnist.validation import ValidationError, validate_eval_config, validate_train_config
from swnist.webapp import inference

TRAINERS = {"features_cnn": run_training}


class JobManager:
    """Lanza entrenamientos y evaluaciones validados, y permite detenerlos."""

    def __init__(
        self,
        registry: ExperimentRegistry | None = None,
        *,
        evaluations: EvaluationRegistry | None = None,
        backups_dir: Path | str | None = None,
    ) -> None:
        self.registry = registry or ExperimentRegistry()
        self.evaluations = evaluations or EvaluationRegistry()
        self.backups_dir = backups_dir
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # --- helpers -----------------------------------------------------------

    def _spawn(self, job_id: str, target) -> None:
        stop_event = threading.Event()

        def job() -> None:
            try:
                target(stop_event)
            except Exception:  # noqa: BLE001 - el estado ya quedó en 'failed'
                traceback.print_exc()
            finally:
                with self._lock:
                    self._jobs.pop(job_id, None)

        thread = threading.Thread(target=job, name=f"job-{job_id}", daemon=True)
        with self._lock:
            self._jobs[job_id] = {"thread": thread, "stop": stop_event}
        thread.start()

    # --- entrenamiento -----------------------------------------------------

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Valida ANTES de crear nada (regla 13). Lanza ValidationError con razón."""
        init_from = (config or {}).get("init_from") or None
        source = None
        if init_from is not None and self.registry.exists(init_from):
            source = self.registry.get(init_from)
        return validate_train_config(config, source_experiment=source)

    def start(self, config: dict[str, Any], *, label: str | None = None) -> str:
        clean = self.validate(config)
        exp_id = self.registry.create(clean, label=label)
        trainer = TRAINERS[clean["nn"]]

        def target(stop_event: threading.Event) -> None:
            try:
                trainer(
                    clean,
                    exp_id,
                    self.registry,
                    stop_event=stop_event,
                    backups_dir=self.backups_dir,
                )
            finally:
                inference.clear_caches()

        self._spawn(exp_id, target)
        return exp_id

    # --- evaluación --------------------------------------------------------

    def validate_evaluation(self, config: dict[str, Any]) -> dict[str, Any]:
        exp_id = (config or {}).get("experiment")
        experiment = (
            self.registry.get(exp_id)
            if exp_id is not None and self.registry.exists(exp_id)
            else None
        )
        return validate_eval_config(config, experiment=experiment)

    def start_evaluation(self, config: dict[str, Any], *, label: str | None = None) -> str:
        clean = self.validate_evaluation(config)
        eval_id = self.evaluations.create(clean, label=label)

        def target(stop_event: threading.Event) -> None:
            run_evaluation(
                clean,
                eval_id,
                self.evaluations,
                self.registry,
                stop_event=stop_event,
            )

        self._spawn(eval_id, target)
        return eval_id

    # --- ciclo de vida -----------------------------------------------------

    def stop(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValidationError(
                f"'{job_id}' no está corriendo, no hay nada que detener."
            )
        job["stop"].set()

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def running_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._jobs)

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        """Espera a que termine un job (usado por los tests)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            job["thread"].join(timeout)
