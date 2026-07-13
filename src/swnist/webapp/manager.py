"""Jobs de entrenamiento en hilos, con parada cooperativa."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from swnist.experiments.registry import ExperimentRegistry
from swnist.training.train_features_cnn import run_training
from swnist.validation import ValidationError, validate_train_config
from swnist.webapp import inference

TRAINERS = {"features_cnn": run_training}


class TrainingManager:
    """Lanza entrenamientos validados y permite detenerlos."""

    def __init__(
        self,
        registry: ExperimentRegistry | None = None,
        *,
        backups_dir: Path | str | None = None,
    ) -> None:
        self.registry = registry or ExperimentRegistry()
        self.backups_dir = backups_dir
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # --- validación --------------------------------------------------------

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Valida ANTES de crear nada (regla 13). Lanza ValidationError con razón."""
        init_from = (config or {}).get("init_from") or None
        source = None
        if init_from is not None and self.registry.exists(init_from):
            source = self.registry.get(init_from)
        return validate_train_config(config, source_experiment=source)

    # --- ciclo de vida -----------------------------------------------------

    def start(self, config: dict[str, Any], *, label: str | None = None) -> str:
        clean = self.validate(config)
        exp_id = self.registry.create(clean, label=label)

        stop_event = threading.Event()
        trainer = TRAINERS[clean["nn"]]

        def job() -> None:
            try:
                trainer(
                    clean,
                    exp_id,
                    self.registry,
                    stop_event=stop_event,
                    backups_dir=self.backups_dir,
                )
            except Exception:  # noqa: BLE001 - el estado ya quedó en 'failed'
                traceback.print_exc()
            finally:
                inference.clear_caches()
                with self._lock:
                    self._jobs.pop(exp_id, None)

        thread = threading.Thread(target=job, name=f"train-{exp_id}", daemon=True)
        with self._lock:
            self._jobs[exp_id] = {"thread": thread, "stop": stop_event}
        thread.start()
        return exp_id

    def stop(self, exp_id: str) -> None:
        with self._lock:
            job = self._jobs.get(exp_id)
        if job is None:
            raise ValidationError(
                f"el experimento '{exp_id}' no está corriendo, no hay nada que detener."
            )
        job["stop"].set()

    def is_running(self, exp_id: str) -> bool:
        with self._lock:
            return exp_id in self._jobs

    def running_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._jobs)

    def wait(self, exp_id: str, timeout: float | None = None) -> None:
        """Espera a que termine un entrenamiento (usado por los tests)."""
        with self._lock:
            job = self._jobs.get(exp_id)
        if job is not None:
            job["thread"].join(timeout)
