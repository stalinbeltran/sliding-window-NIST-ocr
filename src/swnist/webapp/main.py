"""Web app de entrenamiento: API REST + frontend estático.

Correr con: python scripts/run_webapp.py  →  http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from swnist.data.registry import list_datasets
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import list_nns
from .manager import TrainingManager

app = FastAPI(title="sliding-window-NIST-ocr")
registry = ExperimentRegistry()
manager = TrainingManager(registry)


@app.middleware("http")
async def no_cache(request, call_next):
    """Evita que el navegador sirva frontend viejo tras actualizar el código."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response

STATIC_DIR = Path(__file__).parent / "static"


class TrainRequest(BaseModel):
    nn: str
    config: dict


@app.get("/api/nns")
def api_nns():
    return list_nns()


@app.get("/api/datasets")
def api_datasets(nn: str | None = None):
    return list_datasets(nn)


@app.get("/api/experiments")
def api_experiments(nn: str | None = None, status: str | None = None):
    exps = registry.list_experiments(nn=nn, status=status)
    for e in exps:
        e["running"] = manager.is_running(e["id"])
    return exps


@app.get("/api/experiments/{exp_id}")
def api_experiment(exp_id: str):
    try:
        info = registry.get_experiment(exp_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Experimento no encontrado: {exp_id}")
    info["running"] = manager.is_running(exp_id)
    return info


@app.get("/api/experiments/{exp_id}/metrics")
def api_metrics(exp_id: str):
    return registry.get_metrics(exp_id)


@app.post("/api/train")
def api_train(req: TrainRequest):
    try:
        exp_id = manager.start(req.nn, req.config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"experiment_id": exp_id}


@app.post("/api/experiments/{exp_id}/stop")
def api_stop(exp_id: str):
    return {"stopped": manager.stop(exp_id)}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
