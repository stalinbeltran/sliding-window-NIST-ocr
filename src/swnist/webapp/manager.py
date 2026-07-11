"""Gestor de trabajos en segundo plano (hilos) para la web app: entrenamientos
y evaluaciones, ambos con stop cooperativo."""

import threading
import traceback

from swnist.evaluation.registry import EvaluationRegistry
from swnist.evaluation.runner import run_evaluation
from swnist.experiments.registry import ExperimentRegistry
from swnist.training import train_dimensionador, train_secuenciador
from swnist.validation import validate_eval_config, validate_train_config

TRAINERS = {
    "dimensionador": train_dimensionador.run_training,
    "secuenciador": train_secuenciador.run_training,
}


class TrainingManager:
    def __init__(self, registry: ExperimentRegistry | None = None,
                 eval_registry: EvaluationRegistry | None = None):
        self.registry = registry or ExperimentRegistry()
        self.eval_registry = eval_registry or EvaluationRegistry()
        self.jobs: dict[str, dict] = {}  # exp_id/eval_id -> {"thread", "stop_event"}

    def _launch(self, job_id: str, target) -> None:
        stop_event = threading.Event()
        thread = threading.Thread(target=target, args=(stop_event,), daemon=True,
                                  name=f"job-{job_id}")
        self.jobs[job_id] = {"thread": thread, "stop_event": stop_event}
        thread.start()

    def start(self, nn: str, config: dict) -> str:
        if nn not in TRAINERS:
            raise ValueError(f"NN desconocida: {nn!r}")
        # Valida compatibilidad antes de crear el experimento; ValueError → 400
        config = validate_train_config(nn, config, self.registry)
        exp_id = self.registry.create_experiment(nn, config)

        def _run(stop_event):
            try:
                TRAINERS[nn](exp_id, config, self.registry, stop_event)
            except Exception:
                self.registry.mark_finished(exp_id, status="failed",
                                            error=traceback.format_exc())

        self._launch(exp_id, _run)
        return exp_id

    def start_evaluation(self, config: dict) -> str:
        config = validate_eval_config(config, self.registry)
        eval_id = self.eval_registry.create_evaluation(config)

        def _run(stop_event):
            try:
                run_evaluation(eval_id, config, self.eval_registry, self.registry,
                               stop_event)
            except Exception:
                self.eval_registry.update_status(eval_id, status="failed",
                                                 error=traceback.format_exc())

        self._launch(eval_id, _run)
        return eval_id

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job["thread"].is_alive():
            job["stop_event"].set()
            return True
        return False

    def is_running(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        return bool(job and job["thread"].is_alive())
