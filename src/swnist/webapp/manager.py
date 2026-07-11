"""Gestor de trabajos de entrenamiento en segundo plano (hilos) para la web app."""

import threading
import traceback

from swnist.experiments.registry import ExperimentRegistry
from swnist.training import train_dimensionador, train_secuenciador

TRAINERS = {
    "dimensionador": train_dimensionador.run_training,
    "secuenciador": train_secuenciador.run_training,
}


class TrainingManager:
    def __init__(self, registry: ExperimentRegistry | None = None):
        self.registry = registry or ExperimentRegistry()
        self.jobs: dict[str, dict] = {}  # exp_id -> {"thread", "stop_event"}

    def start(self, nn: str, config: dict) -> str:
        if nn not in TRAINERS:
            raise ValueError(f"NN desconocida: {nn!r}")
        exp_id = self.registry.create_experiment(nn, config)
        stop_event = threading.Event()

        def _run():
            try:
                TRAINERS[nn](exp_id, config, self.registry, stop_event)
            except Exception:
                self.registry.mark_finished(exp_id, status="failed",
                                            error=traceback.format_exc())

        thread = threading.Thread(target=_run, daemon=True, name=f"train-{exp_id}")
        self.jobs[exp_id] = {"thread": thread, "stop_event": stop_event}
        thread.start()
        return exp_id

    def stop(self, exp_id: str) -> bool:
        job = self.jobs.get(exp_id)
        if job and job["thread"].is_alive():
            job["stop_event"].set()
            return True
        return False

    def is_running(self, exp_id: str) -> bool:
        job = self.jobs.get(exp_id)
        return bool(job and job["thread"].is_alive())
