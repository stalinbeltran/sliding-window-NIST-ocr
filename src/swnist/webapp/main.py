"""Web app: API REST + estáticos. Todo entrenamiento se lanza desde aquí (regla 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from swnist.data import registry as data_registry
from swnist.data.registry import dataset_info, list_datasets
from swnist.evaluation.registry import EvaluationRegistry
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import list_nns
from swnist.validation import ValidationError, validate_filter
from swnist.webapp import inference
from swnist.webapp.manager import JobManager

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    experiments_dir: Path | str | None = None,
    backups_dir: Path | str | None = None,
    evaluations_dir: Path | str | None = None,
    custom_datasets_dir: Path | str | None = None,
) -> FastAPI:
    registry = ExperimentRegistry(experiments_dir)
    evaluations = EvaluationRegistry(evaluations_dir)
    datasets = data_registry.configure(custom_datasets_dir)
    manager = JobManager(registry, evaluations=evaluations, backups_dir=backups_dir)

    app = FastAPI(title="sliding-window-NIST-ocr")
    app.state.registry = registry
    app.state.evaluations = evaluations
    app.state.datasets = datasets
    app.state.manager = manager

    def _experiment(exp_id: str) -> dict[str, Any]:
        try:
            exp = registry.get(exp_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        exp["running"] = manager.is_running(exp_id)
        return exp

    def _evaluation(eval_id: str) -> dict[str, Any]:
        try:
            evaluation = evaluations.get(eval_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        evaluation["running"] = manager.is_running(eval_id)
        return evaluation

    # --- catálogos ---------------------------------------------------------

    @app.get("/api/nns")
    def get_nns() -> list[dict[str, Any]]:
        return list_nns()

    @app.get("/api/datasets")
    def get_datasets(nn: str | None = Query(default=None)) -> list[dict[str, Any]]:
        return list_datasets(nn)

    # --- experimentos ------------------------------------------------------

    @app.get("/api/experiments")
    def get_experiments() -> list[dict[str, Any]]:
        running = set(manager.running_ids())
        items = registry.list()
        for item in items:
            item["running"] = item["id"] in running
        return items

    @app.post("/api/experiments", status_code=201)
    def post_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        config = payload.get("config") or payload
        try:
            exp_id = manager.start(config, label=payload.get("label"))
        except ValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _experiment(exp_id)

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str) -> dict[str, Any]:
        return _experiment(exp_id)

    @app.get("/api/experiments/{exp_id}/metrics")
    def get_metrics(exp_id: str, since: int = Query(default=0, ge=0)) -> dict[str, Any]:
        exp = _experiment(exp_id)
        records = registry.metrics(exp_id, since=since)
        return {
            "id": exp_id,
            "status": exp["status"],
            "running": exp["running"],
            "since": since,
            "records": records,
            "next": since + len(records),
        }

    @app.post("/api/experiments/{exp_id}/stop")
    def post_stop(exp_id: str) -> dict[str, Any]:
        _experiment(exp_id)
        try:
            manager.stop(exp_id)
        except ValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"stopping": exp_id}

    @app.patch("/api/experiments/{exp_id}")
    def patch_experiment(exp_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        _experiment(exp_id)
        try:
            registry.rename(exp_id, payload.get("label", ""))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _experiment(exp_id)

    @app.post("/api/experiments/{exp_id}/duplicate", status_code=201)
    def post_duplicate(exp_id: str) -> dict[str, Any]:
        _experiment(exp_id)
        return _experiment(registry.duplicate(exp_id))

    @app.delete("/api/experiments/{exp_id}")
    def delete_experiment(exp_id: str) -> dict[str, Any]:
        _experiment(exp_id)
        if manager.is_running(exp_id):
            raise HTTPException(
                409, f"el experimento '{exp_id}' está corriendo. Deténlo antes de eliminarlo."
            )
        used_by = [e["id"] for e in evaluations.list(experiment=exp_id)]
        if used_by:
            raise HTTPException(
                409,
                f"no se puede eliminar '{exp_id}': tiene {len(used_by)} evaluación(es) "
                f"({used_by[:3]}...). Elimínalas primero.",
            )
        try:
            registry.delete(exp_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        inference.clear_caches()
        return {"deleted": exp_id}

    # --- evaluaciones ------------------------------------------------------

    @app.get("/api/evaluations")
    def get_evaluations(
        experiment: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        running = set(manager.running_ids())
        items = evaluations.list(experiment=experiment)
        for item in items:
            item["running"] = item["id"] in running
        return items

    @app.post("/api/evaluations", status_code=201)
    def post_evaluation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        config = payload.get("config") or payload
        try:
            eval_id = manager.start_evaluation(config, label=payload.get("label"))
        except ValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _evaluation(eval_id)

    @app.get("/api/evaluations/{eval_id}")
    def get_evaluation(eval_id: str) -> dict[str, Any]:
        return _evaluation(eval_id)

    @app.get("/api/evaluations/{eval_id}/results")
    def get_results(
        eval_id: str,
        outcome: str = Query(default="all"),
        label: int | None = Query(default=None),
        pred: int | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=48, ge=1, le=200),
        thumbs: bool = Query(default=True),
    ) -> dict[str, Any]:
        evaluation = _evaluation(eval_id)
        try:
            filtro = validate_filter(
                {"outcome": outcome, "label": label, "pred": pred},
                num_classes=evaluation.get("num_classes", 10),
            )
        except ValidationError as exc:
            raise HTTPException(400, str(exc)) from exc

        payload = evaluations.results(eval_id, filtro=filtro, offset=offset, limit=limit)
        if thumbs:
            config = evaluation["config"]
            for result in payload["results"]:
                result["thumb"] = inference.sample_thumb(
                    config["dataset"], split=config["split"], index=result["index"]
                )
        return payload

    @app.patch("/api/evaluations/{eval_id}")
    def patch_evaluation(eval_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        _evaluation(eval_id)
        try:
            evaluations.rename(eval_id, payload.get("label", ""))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _evaluation(eval_id)

    @app.delete("/api/evaluations/{eval_id}")
    def delete_evaluation(eval_id: str) -> dict[str, Any]:
        _evaluation(eval_id)
        try:
            evaluations.delete(eval_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": eval_id}

    # --- datasets custom (CRUD) -------------------------------------------

    def _dataset_users(name: str) -> dict[str, list[str]]:
        return {
            "experiments": [
                exp["id"]
                for exp in registry.list()
                if exp["config"]["dataset"]["name"] == name
            ],
            "evaluations": [
                ev["id"] for ev in evaluations.list() if ev["dataset"] == name
            ],
        }

    def _describe_users(users: dict[str, list[str]]) -> str:
        """'2 experimento(s) y 1 evaluación(es)' → solo lo que de verdad lo usa."""
        parts = []
        if users["experiments"]:
            parts.append(f"{len(users['experiments'])} experimento(s) {users['experiments'][:3]}")
        if users["evaluations"]:
            parts.append(f"{len(users['evaluations'])} evaluación(es) {users['evaluations'][:3]}")
        return " y ".join(parts)

    @app.get("/api/custom-datasets")
    def get_custom_datasets() -> list[dict[str, Any]]:
        return [
            {**definition, "users": _dataset_users(definition["name"])}
            for definition in datasets.list()
        ]

    @app.post("/api/custom-datasets", status_code=201)
    def post_custom_dataset(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Crea un dataset a partir del filtro de una evaluación, o recortando un base."""
        name = payload.get("name")
        from_evaluation = payload.get("from_evaluation")
        from_base = payload.get("from_base")

        if from_evaluation:
            eval_id = from_evaluation.get("evaluation")
            evaluation = _evaluation(eval_id)
            if evaluation["status"] != "completed":
                raise HTTPException(
                    400,
                    f"la evaluación '{eval_id}' está en estado '{evaluation['status']}': "
                    "solo se puede guardar el filtro de una evaluación completada.",
                )
            try:
                filtro = validate_filter(
                    from_evaluation, num_classes=evaluation.get("num_classes", 10)
                )
            except ValidationError as exc:
                raise HTTPException(400, str(exc)) from exc

            indices = evaluations.matching_indices(eval_id, filtro)
            origin = {
                "type": "evaluation",
                "evaluation": eval_id,
                "experiment": evaluation["experiment"],
                "dataset": evaluation["dataset"],
                "filter": filtro,
            }
            base = evaluation["base"]
            split = evaluation["base_split"]
        elif from_base:
            base = from_base.get("base")
            split = from_base.get("split", "test")
            limit = from_base.get("limit")
            try:
                info = dataset_info(base)
            except KeyError as exc:
                raise HTTPException(400, str(exc)) from exc
            if info["is_custom"]:
                raise HTTPException(
                    400,
                    f"'{base}' ya es un dataset custom. Para derivar de él, evalúalo y "
                    "guarda el filtro de esa evaluación.",
                )
            total = len(inference.get_dataset(base, train=(split == "train")))
            count = total if limit is None else min(int(limit), total)
            indices = list(range(count))
            origin = {"type": "base", "base": base, "split": split, "limit": limit}
        else:
            raise HTTPException(
                400,
                "hay que indicar 'from_evaluation' (guardar un filtro) o 'from_base' "
                "(recortar un dataset base).",
            )

        try:
            definition = datasets.create(
                name, base=base, split=split, indices=indices, origin=origin
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        inference.clear_caches()
        return {**definition, "users": _dataset_users(definition["name"])}

    @app.get("/api/custom-datasets/{name}")
    def get_custom_dataset(name: str) -> dict[str, Any]:
        try:
            definition = datasets.get(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {**definition, "users": _dataset_users(name)}

    @app.patch("/api/custom-datasets/{name}")
    def patch_custom_dataset(name: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            datasets.get(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        users = _dataset_users(name)
        if users["experiments"] or users["evaluations"]:
            raise HTTPException(
                409,
                f"no se puede renombrar '{name}': lo referencian por nombre "
                f"{_describe_users(users)}.",
            )
        try:
            definition = datasets.rename(name, payload.get("name", ""))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        inference.clear_caches()
        return {**definition, "users": _dataset_users(definition["name"])}

    @app.delete("/api/custom-datasets/{name}")
    def delete_custom_dataset(name: str) -> dict[str, Any]:
        try:
            datasets.get(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        users = _dataset_users(name)
        if users["experiments"] or users["evaluations"]:
            raise HTTPException(
                409,
                f"no se puede eliminar '{name}': lo usan {_describe_users(users)}. "
                "Elimínalos primero.",
            )
        datasets.delete(name)
        inference.clear_caches()
        return {"deleted": name}

    # --- features / inferencia --------------------------------------------

    @app.get("/api/datasets/{name}/samples")
    def get_samples(
        name: str,
        split: str = Query(default="test", pattern="^(train|test)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=24, ge=1, le=200),
    ) -> dict[str, Any]:
        try:
            return inference.dataset_samples(name, split=split, offset=offset, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/predict")
    def post_predict(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        exp_id = payload.get("experiment")
        exp = _experiment(exp_id)
        dataset = payload.get("dataset") or exp["config"]["dataset"]["name"]
        try:
            return inference.predict(
                exp_id,
                registry,
                dataset=dataset,
                split=payload.get("split", "test"),
                index=int(payload.get("index", 0)),
                max_maps=int(payload.get("max_maps", inference.MAX_MAPS_PER_LAYER)),
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/experiments/{exp_id}/kernels")
    def get_kernels(exp_id: str) -> dict[str, Any]:
        _experiment(exp_id)
        try:
            return inference.kernels(exp_id, registry)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # --- estáticos ---------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
