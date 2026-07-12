"""Inferencia interactiva para la web app: miniaturas de muestras, predicción
sobre una muestra concreta y trace paso a paso de la ventana deslizante.

Cachea datasets y modelos en memoria (claves inmutables) porque cada petición
de la UI toca pocas muestras; los modelos ya entrenados no cambian (cada
entrenamiento crea un experimento nuevo).
"""

import base64
import io
import json

import torch
import torch.nn.functional as F
from PIL import Image

from swnist.data.datasets import IMAGE_SIZE
from swnist.data.registry import build_dataset, dataset_info
from swnist.data.windows import extract_window, grid_positions
from swnist.evaluation.runner import load_model
from swnist.experiments.registry import ExperimentRegistry
from swnist.training.common import get_device
from swnist.validation import sequence_effective_params

# Estadísticas de normalización de MNIST (para volver a imagen visible)
_MEAN, _STD = 0.1307, 0.3081

_dataset_cache: dict[tuple, object] = {}
_model_cache: dict[str, tuple] = {}


def clear_caches() -> None:
    """Invalidación total (p. ej. al borrar/renombrar datasets custom)."""
    _dataset_cache.clear()
    _model_cache.clear()


def get_dataset(name: str, params: dict, train: bool, seed: int):
    key = (name, json.dumps(params, sort_keys=True), train, seed)
    if key not in _dataset_cache:
        _dataset_cache[key] = build_dataset(name, params, train=train, seed=seed)
    return _dataset_cache[key]


def get_model(exp_registry: ExperimentRegistry, exp_id: str):
    if exp_id not in _model_cache:
        _model_cache[exp_id] = load_model(exp_registry, exp_id, get_device())
    return _model_cache[exp_id]


def tensor_to_png_b64(img: torch.Tensor) -> str:
    """img (1, H, W) normalizada -> PNG base64 (escala de grises)."""
    x = (img.squeeze(0) * _STD + _MEAN).clamp(0, 1).mul(255).byte().cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(x, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_samples(name: str, params: dict, split: str, seed: int,
                offset: int, limit: int, indices: list[int] | None = None,
                label: int | None = None) -> dict:
    """Página de miniaturas: {total, items: [{index, label, png}]}."""
    ds = get_dataset(name, params, train=(split == "train"), seed=seed)
    if indices is None:
        pool = range(len(ds))
    else:
        pool = [i for i in indices if 0 <= i < len(ds)]
    if label is not None:
        pool = [i for i in pool if ds.display_item(i)[1] == label]
    total = len(pool)
    page = list(pool)[offset:offset + limit]
    items = []
    for i in page:
        img, y = ds.display_item(i)
        items.append({"index": i, "label": int(y), "png": tensor_to_png_b64(img)})
    return {"total": total, "items": items}


@torch.no_grad()
def predict_sample(exp_registry: ExperimentRegistry, exp_id: str, dataset_cfg: dict,
                   split: str, index: int, seed: int = 42, stride: int = 7) -> dict:
    """Aplica la NN del experimento a una muestra y devuelve predicción + trace.

    El trace (deslizamiento paso a paso) se incluye cuando la muestra es una
    imagen completa: para el secuenciador es su recorrido real; para el
    dimensionador se desliza su ventana (window_size del modelo) con `stride`.
    """
    device = get_device()
    nn_name, model, exp_cfg = get_model(exp_registry, exp_id)

    ds_name = dataset_cfg["name"]
    ds_params = dict(dataset_cfg.get("params") or {})
    info = dataset_info(ds_name)

    if nn_name == "secuenciador":
        # Ventana Y stride: los efectivos del entrenamiento (la trayectoria es
        # entrada de la red); un stride distinto en la petición se ignora.
        ds_params.update(sequence_effective_params(exp_cfg, exp_registry))

    ds = get_dataset(ds_name, ds_params, train=(split == "train"), seed=seed)
    if not (0 <= index < len(ds)):
        raise ValueError(f"Índice fuera de rango: {index} (el dataset tiene {len(ds)} muestras).")
    display_img, label = ds.display_item(index)

    out = {
        "experiment": exp_id, "nn": nn_name, "index": index, "label": int(label),
        "image_png": tensor_to_png_b64(display_img),
        "image_size": int(display_img.shape[-1]),
        "trace": None, "trace_reason": None,
    }

    if nn_name == "dimensionador":
        item, _ = ds[index]
        probs = F.softmax(model(item.unsqueeze(0).to(device)), dim=-1)[0]
        out.update(_pred_fields(probs))
        ws = int(model.config["window_size"])
        if info["full_image"] and ws < IMAGE_SIZE:
            positions = grid_positions(IMAGE_SIZE, ws, stride)
            steps = []
            for top, left in positions:
                w = extract_window(display_img, top, left, ws)
                p = F.softmax(model(w.unsqueeze(0).to(device)), dim=-1)[0]
                steps.append(_step_fields(p))
            out["trace"] = {"window_size": ws, "stride": stride, "positions": positions,
                            "steps": steps, "kind": "dimensionador"}
        else:
            out["trace_reason"] = (
                "La muestra ya es una ventana única: no hay deslizamiento que "
                "animar. Usa un dataset de imagen completa (p. ej. mnist_full) "
                "para ver el recorrido." if not info["full_image"] else
                "La ventana del modelo cubre toda la imagen: no hay deslizamiento.")
    else:
        windows, positions_norm, _ = ds[index]
        _, dim_model, _ = get_model(exp_registry, exp_cfg["dimensionador_experiment"])
        feats = dim_model.features(windows.to(device))
        logits_seq, _ = model(feats.unsqueeze(0), positions_norm.unsqueeze(0).to(device))
        probs_seq = F.softmax(logits_seq[0], dim=-1)  # (T, C)
        out.update(_pred_fields(probs_seq[-1]))
        ws = int(dim_model.config["window_size"])
        # Recorrido real de la muestra: fijo (raster) o derivado del trazo (contour)
        traj = getattr(ds, "trajectory", None)
        positions = traj(index) if traj is not None else []
        out["trace"] = {
            "window_size": ws,
            "stride": ds_params.get("stride"),
            "num_steps": ds_params.get("num_steps"),
            "positions": [list(p) for p in positions],
            "steps": [_step_fields(p) for p in probs_seq],
            "kind": "secuenciador",
        }
    return out


def _pred_fields(probs: torch.Tensor) -> dict:
    top2 = probs.topk(2)
    return {
        "pred": int(top2.indices[0]),
        "conf": round(float(top2.values[0]), 4),
        "margin": round(float(top2.values[0] - top2.values[1]), 4),
        "probs": [round(float(p), 4) for p in probs],
    }


def _step_fields(probs: torch.Tensor) -> dict:
    return {"pred": int(probs.argmax()), "probs": [round(float(p), 4) for p in probs]}
