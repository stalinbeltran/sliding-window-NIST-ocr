"""Web app: API REST + frontend estático.

Correr con: python scripts/run_webapp.py  →  http://127.0.0.1:8000

Además de entrenar (POST /api/train, con re-entrenamiento vía config.init_from),
la API permite: evaluar una NN entrenada sobre cualquier dataset compatible,
inspeccionar muestras (miniaturas, predicción y trace de la ventana deslizante),
filtrar resultados (aciertos/fallos/ambiguos) y guardarlos como dataset custom,
y CRUD de experimentos y datasets custom. Toda restricción de compatibilidad
responde 400 con la razón.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from swnist.data import registry as data_registry
from swnist.data.custom import slugify
from swnist.data.datasets import IMAGE_SIZE
from swnist.data.registry import list_datasets
from swnist.data.windows import grid_positions
from swnist.evaluation.registry import EvaluationRegistry
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import list_nns
from swnist.validation import sequence_effective_params
from . import inference
from .manager import TrainingManager

app = FastAPI(title="sliding-window-NIST-ocr")
registry = ExperimentRegistry()
eval_registry = EvaluationRegistry()
manager = TrainingManager(registry, eval_registry)


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


class EvalRequest(BaseModel):
    experiment: str
    dataset: dict            # {"name": ..., "params": {...}}
    split: str = "test"
    limit: int | None = None
    batch_size: int = 256
    seed: int = 42


class PredictRequest(BaseModel):
    experiment: str
    dataset: dict
    split: str = "test"
    index: int
    seed: int = 42
    stride: int = 7          # paso del trace del dimensionador


class SaveDatasetRequest(BaseModel):
    result: str = "all"      # all | correct | failed | ambiguous
    ambiguity: float = 0.2
    label: int | None = None
    pred: int | None = None
    name: str | None = None
    description: str = ""


class CustomDatasetPatch(BaseModel):
    new_name: str | None = None
    description: str | None = None


class LabelPatch(BaseModel):
    label: str


def _400(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError) as e:
        detail = e.args[0] if e.args else str(e)
        raise HTTPException(400, str(detail))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# ---------- catálogos ----------

@app.get("/api/nns")
def api_nns():
    return list_nns()


@app.get("/api/datasets")
def api_datasets(nn: str | None = None):
    return list_datasets(nn)


@app.get("/api/datasets/{name}/slide")
def api_dataset_slide(name: str, window_size: int | None = None,
                      stride: int | None = None):
    """Recorrido de la ventana deslizante de un dataset, para visualizarlo y
    verificar el stride sin tener que lanzar una prueba. window_size/stride
    sobrescriben los efectivos del dataset (solo para la visualización)."""
    info = _400(data_registry.dataset_info, name)
    eff = _400(data_registry.effective_params, name, {})
    uses_stride = "stride" in eff
    ws = int(window_size) if window_size is not None else int(eff.get("window_size", IMAGE_SIZE))
    st = int(stride) if stride is not None else int(eff.get("stride", 7))
    if not 1 <= ws <= IMAGE_SIZE:
        raise HTTPException(400, f"window_size debe estar entre 1 y {IMAGE_SIZE}; "
                                 f"recibido: {ws}.")
    if st < 1:
        raise HTTPException(400, f"stride debe ser ≥ 1; recibido: {st}.")
    positions = grid_positions(IMAGE_SIZE, ws, st) if ws < IMAGE_SIZE else [(0, 0)]
    axis = sorted({p[0] for p in positions})
    notes = []
    if not uses_stride:
        notes.append("Este dataset no desliza una ventana: muestrea ventanas "
                     "aleatorias por imagen (windows_per_image), así que no tiene "
                     "stride propio. El recorrido mostrado es el raster hipotético "
                     "que usa, p. ej., el trace del dimensionador en Probar.")
    if ws >= IMAGE_SIZE:
        notes.append("La ventana cubre toda la imagen: el recorrido colapsa a 1 paso.")
    elif st > ws:
        notes.append(f"stride ({st}) > ventana ({ws}): quedan franjas de hasta "
                     f"{st - ws} px sin ver entre posiciones consecutivas (solo el "
                     f"borde final se cubre siempre).")
    elif st < ws:
        notes.append(f"Ventanas solapadas: {ws - st} px de solape entre posiciones "
                     f"consecutivas.")
    return {
        "name": name, "image_size": IMAGE_SIZE, "window_size": ws, "stride": st,
        "dataset_uses_stride": uses_stride, "effective_defaults": eff,
        "full_image": info.get("full_image", True),
        "steps": len(positions), "axis_coords": axis,
        "positions": [list(p) for p in positions],
        "note": " ".join(notes),
    }


@app.get("/api/datasets/{name}/samples")
def api_dataset_samples(name: str, split: str = "train", seed: int = 42,
                        offset: int = 0, limit: int = 60,
                        label: int | None = None, params: str = "{}"):
    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "params debe ser JSON válido")
    return _400(inference.get_samples, name, p, split, seed, offset,
                min(limit, 200), None, label)


# ---------- experimentos (CRUD) ----------

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


@app.patch("/api/experiments/{exp_id}")
def api_experiment_label(exp_id: str, req: LabelPatch):
    return _400(registry.set_label, exp_id, req.label)


@app.post("/api/experiments/{exp_id}/copy")
def api_experiment_copy(exp_id: str):
    if manager.is_running(exp_id):
        raise HTTPException(409, "El experimento está corriendo: espera a que "
                                 "termine antes de duplicarlo.")
    return {"experiment_id": _400(registry.copy_experiment, exp_id)}


@app.delete("/api/experiments/{exp_id}")
def api_experiment_delete(exp_id: str):
    if manager.is_running(exp_id):
        raise HTTPException(409, "El experimento está corriendo: detenlo antes "
                                 "de eliminarlo.")
    refs = registry.referencing_experiments(exp_id)
    if refs:
        raise HTTPException(409,
            f"No se puede eliminar {exp_id}: los secuenciadores {refs} lo usan "
            f"como dimensionador (lo necesitan para evaluar y re-entrenar). "
            f"Elimina primero esos experimentos.")
    _400(registry.delete_experiment, exp_id)
    inference.clear_caches()
    return {"deleted": exp_id}


# ---------- entrenamiento ----------

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


# ---------- evaluaciones ----------

@app.post("/api/evaluations")
def api_eval_start(req: EvalRequest):
    try:
        eval_id = manager.start_evaluation(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"evaluation_id": eval_id}


@app.get("/api/evaluations")
def api_evaluations(experiment: str | None = None):
    evals = eval_registry.list_evaluations(experiment)
    for e in evals:
        e["running"] = manager.is_running(e["id"])
    return evals


@app.get("/api/evaluations/{eval_id}")
def api_evaluation(eval_id: str):
    try:
        info = eval_registry.get_evaluation(eval_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Evaluación no encontrada: {eval_id}")
    info["running"] = manager.is_running(eval_id)
    return info


@app.get("/api/evaluations/{eval_id}/results")
def api_eval_results(eval_id: str, result: str = "all", ambiguity: float = 0.2,
                     label: int | None = None, pred: int | None = None,
                     offset: int = 0, limit: int = 60, with_images: bool = True):
    try:
        info = eval_registry.get_evaluation(eval_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Evaluación no encontrada: {eval_id}")
    rows = _400(eval_registry.filter_results, eval_id, result, label, pred, ambiguity)
    total = len(rows)
    page = rows[offset:offset + min(limit, 200)]
    if with_images and page:
        cfg = info["config"]
        ds_params = _eval_ds_params(cfg)
        imgs = _400(inference.get_samples, cfg["dataset"]["name"], ds_params,
                    cfg["split"], cfg.get("seed", 42), 0, len(page),
                    [r["index"] for r in page], None)
        by_idx = {it["index"]: it["png"] for it in imgs["items"]}
        for r in page:
            r["png"] = by_idx.get(r["index"])
    return {"total": total, "items": page, "summary": info["status"].get("summary")}


def _eval_ds_params(cfg: dict) -> dict:
    """Params efectivos del dataset de una evaluación (ventana Y stride del
    entrenamiento del secuenciador) — para que las miniaturas y el predict
    apunten a la misma muestra. Las evaluaciones nuevas ya guardan estos valores
    en su config; esto corrige también las antiguas."""
    ds_params = dict(cfg["dataset"].get("params") or {})
    try:
        exp = registry.get_experiment(cfg["experiment"])
        if exp["config"].get("nn") == "secuenciador":
            ds_params.update(sequence_effective_params(exp["config"], registry))
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return ds_params


@app.post("/api/evaluations/{eval_id}/save-dataset")
def api_eval_save_dataset(eval_id: str, req: SaveDatasetRequest):
    try:
        info = eval_registry.get_evaluation(eval_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Evaluación no encontrada: {eval_id}")
    rows = _400(eval_registry.filter_results, eval_id, req.result, req.label,
                req.pred, req.ambiguity)
    indices = [r["index"] for r in rows]
    cfg = info["config"]
    ds_name = cfg["dataset"]["name"]
    ds_info = _400(data_registry.dataset_info, ds_name)
    if ds_info.get("custom"):
        # Aplanar: los índices del custom refieren a su base builtin.
        d = data_registry.custom_store.get(ds_name)
        indices = [d["indices"][i] for i in indices] if cfg["split"] == d["base"]["split"] \
            else indices
        base = dict(d["base"]) if cfg["split"] == d["base"]["split"] else \
            {"name": d["base"]["name"], "params": d["base"]["params"],
             "split": cfg["split"], "seed": cfg.get("seed", 42)}
    else:
        base = {"name": ds_name, "params": cfg["dataset"].get("params") or {},
                "split": cfg["split"], "seed": cfg.get("seed", 42)}
    filt = {k: v for k, v in
            [("result", req.result), ("label", req.label), ("pred", req.pred),
             ("ambiguity", req.ambiguity if req.result == "ambiguous" else None)]
            if v is not None}
    name = req.name
    if not name:
        # Nombre por defecto (modificable): dataset + filtro, con sufijo si ya existe.
        name, n = f"{ds_name}_{req.result}", 2
        while data_registry.custom_store.exists(slugify(name)):
            name = f"{ds_name}_{req.result}_{n}"
            n += 1
    created = _400(data_registry.custom_store.create, base, indices,
                   name, req.description,
                   {"evaluation": eval_id, "filter": filt})
    inference.clear_caches()
    return created


@app.delete("/api/evaluations/{eval_id}")
def api_eval_delete(eval_id: str):
    if manager.is_running(eval_id):
        raise HTTPException(409, "La evaluación está corriendo: detenla antes.")
    try:
        eval_registry.delete(eval_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"deleted": eval_id}


@app.post("/api/evaluations/{eval_id}/stop")
def api_eval_stop(eval_id: str):
    return {"stopped": manager.stop(eval_id)}


# ---------- datasets custom (CRUD) ----------

@app.get("/api/custom-datasets")
def api_custom_list():
    return data_registry.custom_store.list()


def _custom_dataset_refs(name: str) -> list[str]:
    return [e["id"] for e in registry.list_experiments()
            if e["config"].get("dataset", {}).get("name") == name]


@app.patch("/api/custom-datasets/{name}")
def api_custom_patch(name: str, req: CustomDatasetPatch):
    out = None
    if req.new_name and req.new_name != name:
        refs = _custom_dataset_refs(name)
        if refs:
            raise HTTPException(409,
                f"No se puede renombrar {name!r}: los experimentos {refs} lo "
                f"referencian por nombre (se rompería su reproducibilidad). "
                f"Duplícalo con otro nombre si lo necesitas.")
        out = _400(data_registry.custom_store.rename, name, req.new_name)
        name = out["name"]
        inference.clear_caches()
    if req.description is not None:
        out = _400(data_registry.custom_store.update_description, name, req.description)
    return out or _400(data_registry.custom_store.get_summary, name)


@app.post("/api/custom-datasets/{name}/copy")
def api_custom_copy(name: str, new_name: str | None = None):
    return _400(data_registry.custom_store.copy, name, new_name)


@app.delete("/api/custom-datasets/{name}")
def api_custom_delete(name: str):
    refs = _custom_dataset_refs(name)
    if refs:
        raise HTTPException(409,
            f"No se puede eliminar {name!r}: los experimentos {refs} lo usan "
            f"como dataset (se perdería la reproducibilidad de su entrenamiento). "
            f"Elimina primero esos experimentos.")
    _400(data_registry.custom_store.delete, name)
    inference.clear_caches()
    return {"deleted": name}


# ---------- predicción / trace ----------

@app.post("/api/predict")
def api_predict(req: PredictRequest):
    return _400(inference.predict_sample, registry, req.experiment, req.dataset,
                req.split, req.index, req.seed, req.stride)


# ---------- frontend ----------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
