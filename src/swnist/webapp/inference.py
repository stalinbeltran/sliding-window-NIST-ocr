"""Inferencia para la web app: miniaturas, predicción y extracción de features.

Los *features reconocidos* son las activaciones de cada capa convolucional: cada
kernel produce una matriz H×W que se envía como números (representación matricial)
para pintarla en el navegador.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import numpy as np
import torch

from swnist.data.registry import build_dataset, dataset_info
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.features_cnn import FeaturesCNN
from swnist.training.common import load_checkpoint

MAX_MAPS_PER_LAYER = 64

_model_cache: dict[str, tuple[FeaturesCNN, dict[str, Any]]] = {}
_dataset_cache: dict[tuple[str, bool], Any] = {}


def clear_caches() -> None:
    """Invalidar al borrar/renombrar experimentos o datasets."""
    _model_cache.clear()
    _dataset_cache.clear()


# --- PNG (sin dependencias externas) ---------------------------------------


def encode_png(gray: np.ndarray) -> bytes:
    """PNG en escala de grises a partir de una matriz uint8 (H, W)."""
    height, width = gray.shape
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def png_data_uri(image: np.ndarray) -> str:
    """`image` en [0, 1] → data URI base64 lista para un <img>."""
    gray = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return "data:image/png;base64," + base64.b64encode(encode_png(gray)).decode("ascii")


# --- caches ----------------------------------------------------------------


def get_dataset(name: str, *, train: bool) -> Any:
    """Dataset para explorar/predecir: siempre el split completo (el `limit` de la
    config solo recorta el conjunto de entrenamiento)."""
    key = (name, train)
    if key not in _dataset_cache:
        dataset_info(name)  # valida el nombre
        _dataset_cache[key] = build_dataset(name, train=train)
    return _dataset_cache[key]


def get_model(exp_id: str, registry: ExperimentRegistry) -> tuple[FeaturesCNN, dict[str, Any]]:
    if exp_id not in _model_cache:
        path = registry.checkpoint_path(exp_id, "best.pt")
        if not path.exists():
            raise ValueError(
                f"el experimento '{exp_id}' no tiene checkpoint 'best.pt': entrénalo "
                "(al menos una época completa) antes de usarlo."
            )
        checkpoint = load_checkpoint(path)
        config = checkpoint["config"]
        model = FeaturesCNN.from_config(config["model"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        _model_cache[exp_id] = (model, config)
    return _model_cache[exp_id]


# --- matrices --------------------------------------------------------------


def as_matrix(array: np.ndarray, decimals: int = 4) -> list[list[float]]:
    return np.round(array.astype(float), decimals).tolist()


def map_payload(array: np.ndarray, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "matrix": as_matrix(array),
    }


# --- API de inferencia -----------------------------------------------------


def sample_thumb(name: str, *, split: str, index: int) -> str:
    dataset = get_dataset(name, train=(split == "train"))
    return png_data_uri(dataset.display_item(index))


def dataset_samples(
    name: str, *, split: str = "test", offset: int = 0, limit: int = 24
) -> dict[str, Any]:
    dataset = get_dataset(name, train=(split == "train"))
    total = len(dataset)
    offset = max(0, min(offset, total))
    end = min(offset + limit, total)
    samples = []
    for idx in range(offset, end):
        image = dataset.display_item(idx)
        _, label = dataset[idx]
        samples.append({"index": idx, "label": int(label), "thumb": png_data_uri(image)})
    return {"dataset": name, "split": split, "total": total, "offset": offset, "samples": samples}


def predict(
    exp_id: str,
    registry: ExperimentRegistry,
    *,
    dataset: str,
    split: str = "test",
    index: int = 0,
    max_maps: int = MAX_MAPS_PER_LAYER,
) -> dict[str, Any]:
    """Predicción de una muestra + los feature maps de cada capa (matrices)."""
    model, config = get_model(exp_id, registry)
    data = get_dataset(dataset, train=(split == "train"))
    total = len(data)
    if not 0 <= index < total:
        raise ValueError(
            f"índice {index} fuera de rango: el split '{split}' de '{dataset}' tiene "
            f"{total} muestras (0..{total - 1})."
        )

    image, label = data[index]
    batch = image.unsqueeze(0)
    with torch.no_grad():
        maps = model.feature_maps(batch)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    layers = []
    for i, tensor in enumerate(maps):
        activations = tensor.squeeze(0).numpy()  # (kernels, H, W)
        shown = min(activations.shape[0], max_maps)
        layers.append(
            {
                "layer": i + 1,
                "kernels": int(activations.shape[0]),
                "shown": shown,
                "truncated": shown < activations.shape[0],
                "height": int(activations.shape[1]),
                "width": int(activations.shape[2]),
                "spec": config["model"]["layers"][i],
                "maps": [map_payload(activations[k], k) for k in range(shown)],
            }
        )

    ordered = torch.argsort(probs, descending=True)
    top = [{"digit": int(d), "prob": float(probs[d])} for d in ordered[:3]]
    return {
        "experiment": exp_id,
        "dataset": dataset,
        "split": split,
        "index": index,
        "label": int(label),
        "pred": int(ordered[0]),
        "correct": int(ordered[0]) == int(label),
        "probs": [float(p) for p in probs],
        "top": top,
        "margin": float(probs[ordered[0]] - probs[ordered[1]]),
        "image": as_matrix(data.display_item(index), decimals=3),
        "thumb": png_data_uri(data.display_item(index)),
        "layers": layers,
    }


def kernels(exp_id: str, registry: ExperimentRegistry, *, max_maps: int = MAX_MAPS_PER_LAYER) -> dict[str, Any]:
    """Kernels aprendidos por cada capa, como matrices."""
    model, config = get_model(exp_id, registry)
    layers = []
    for i, weight in enumerate(model.kernels()):
        array = weight.numpy()  # (kernels, in_channels, k, k)
        shown = min(array.shape[0], max_maps)
        layers.append(
            {
                "layer": i + 1,
                "kernels": int(array.shape[0]),
                "shown": shown,
                "truncated": shown < array.shape[0],
                "in_channels": int(array.shape[1]),
                "kernel_size": int(array.shape[2]),
                "spec": config["model"]["layers"][i],
                # Se muestra el kernel del primer canal de entrada de cada filtro.
                "maps": [map_payload(array[k, 0], k) for k in range(shown)],
            }
        )
    return {"experiment": exp_id, "layers": layers}
