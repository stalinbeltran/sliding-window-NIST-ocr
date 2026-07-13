"""Web app: API REST + estáticos. Todo entrenamiento se lanza desde aquí (regla 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from swnist.data.registry import list_datasets
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import list_nns
from swnist.validation import ValidationError
from swnist.webapp import inference
from swnist.webapp.manager import TrainingManager

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    experiments_dir: Path | str | None = None,
    backups_dir: Path | str | None = None,
) -> FastAPI:
    registry = ExperimentRegistry(experiments_dir)
    manager = TrainingManager(registry, backups_dir=backups_dir)
    app = FastAPI(title="sliding-window-NIST-ocr")
    app.state.registry = registry
    app.state.manager = manager

    def _experiment(exp_id: str) -> dict[str, Any]:
        try:
            exp = registry.get(exp_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        exp["running"] = manager.is_running(exp_id)
        return exp

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
        label = payload.get("label", "")
        try:
            registry.rename(exp_id, label)
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
        try:
            registry.delete(exp_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        inference.clear_caches()
        return {"deleted": exp_id}

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
