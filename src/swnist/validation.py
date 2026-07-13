"""Validación previa de configs: toda restricción se comprueba ANTES de crear el
experimento y se explica con una razón (la API responde 400 con ese mensaje)."""

from __future__ import annotations

import copy
from typing import Any

from swnist.data.registry import dataset_info, effective_params
from swnist.models.features_cnn import ACTIVATIONS, POOL_TYPES, LayerSpec, spatial_trace
from swnist.nn_registry import NNS

MAX_KERNEL_SIZE = 28
MAX_LAYERS = 8

OUTCOMES = ("all", "correct", "failed", "ambiguous")
DEFAULT_AMBIGUOUS_THRESHOLD = 0.2


class ValidationError(ValueError):
    """Config inválida; el mensaje explica el porqué y cómo corregirlo."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _positive_int(value: Any, field: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"'{field}' debe ser un entero, se recibió {value!r}.",
    )
    _require(value >= minimum, f"'{field}' debe ser >= {minimum}, se recibió {value}.")
    if maximum is not None:
        _require(value <= maximum, f"'{field}' debe ser <= {maximum}, se recibió {value}.")
    return value


def validate_pool(pool: Any, layer_no: int) -> dict[str, Any] | None:
    """El pooling es opcional: None (o ausente) = sin pooling en esa capa."""
    if pool is None:
        return None
    _require(
        isinstance(pool, dict),
        f"el pooling de la capa {layer_no} debe ser null (sin pooling) o un objeto "
        f'{{"type": "max"|"avg", "size": N}}; se recibió {pool!r}.',
    )
    pool_type = pool.get("type", "max")
    _require(
        pool_type in POOL_TYPES,
        f"el pooling de la capa {layer_no} tiene type '{pool_type}'; "
        f"debe ser uno de {list(POOL_TYPES)}.",
    )
    size = _positive_int(pool.get("size", 2), f"layers[{layer_no}].pool.size", minimum=2, maximum=MAX_KERNEL_SIZE)
    stride = pool.get("stride")
    if stride is not None:
        stride = _positive_int(stride, f"layers[{layer_no}].pool.stride")
    return {"type": pool_type, "size": size, "stride": stride}


def validate_layers(layers: Any) -> list[dict[str, Any]]:
    _require(
        isinstance(layers, list) and len(layers) >= 1,
        "'model.layers' debe ser una lista con al menos 1 capa convolucional.",
    )
    _require(
        len(layers) <= MAX_LAYERS,
        f"'model.layers' admite hasta {MAX_LAYERS} capas; se recibieron {len(layers)}.",
    )
    clean: list[dict[str, Any]] = []
    for i, raw in enumerate(layers):
        n = i + 1
        _require(isinstance(raw, dict), f"la capa {n} debe ser un objeto JSON, se recibió {raw!r}.")
        _require("kernels" in raw, f"la capa {n} no declara 'kernels' (número de filtros).")
        _require("kernel_size" in raw, f"la capa {n} no declara 'kernel_size'.")
        activation = raw.get("activation", "relu")
        _require(
            activation in ACTIVATIONS,
            f"la capa {n} usa la activación '{activation}'; debe ser una de "
            f"{sorted(ACTIVATIONS)}.",
        )
        clean.append(
            {
                "kernels": _positive_int(raw["kernels"], f"layers[{n}].kernels", maximum=256),
                "kernel_size": _positive_int(
                    raw["kernel_size"], f"layers[{n}].kernel_size", maximum=MAX_KERNEL_SIZE
                ),
                "stride": _positive_int(raw.get("stride", 1), f"layers[{n}].stride"),
                "padding": _positive_int(
                    raw.get("padding", 0), f"layers[{n}].padding", minimum=0, maximum=MAX_KERNEL_SIZE
                ),
                "activation": activation,
                "pool": validate_pool(raw.get("pool"), n),
            }
        )
    return clean


def validate_model(model_cfg: Any, *, input_size: int, num_classes: int) -> dict[str, Any]:
    _require(isinstance(model_cfg, dict), "'model' debe ser un objeto JSON.")
    clean = {
        "input_size": input_size,
        "in_channels": 1,
        "num_classes": num_classes,
        "hidden_dim": _positive_int(
            model_cfg.get("hidden_dim", 64), "model.hidden_dim", minimum=0, maximum=4096
        ),
        "dropout": model_cfg.get("dropout", 0.0),
        "layers": validate_layers(model_cfg.get("layers")),
    }
    dropout = clean["dropout"]
    _require(
        isinstance(dropout, (int, float))
        and not isinstance(dropout, bool)
        and 0.0 <= float(dropout) < 1.0,
        f"'model.dropout' debe estar en [0, 1), se recibió {dropout!r}.",
    )
    clean["dropout"] = float(dropout)

    # El mapa no puede agotarse: conv y pooling deben caber en el tamaño espacial.
    try:
        spatial_trace([LayerSpec.from_dict(layer) for layer in clean["layers"]], input_size)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return clean


def validate_training(training: Any) -> dict[str, Any]:
    _require(isinstance(training, dict), "'training' debe ser un objeto JSON.")
    clean = {
        "epochs": _positive_int(training.get("epochs", 3), "training.epochs", maximum=1000),
        "batch_size": _positive_int(
            training.get("batch_size", 128), "training.batch_size", maximum=8192
        ),
        "lr": training.get("lr", 1e-3),
        "val_fraction": training.get("val_fraction", 0.1),
        "log_every": _positive_int(training.get("log_every", 50), "training.log_every"),
    }
    lr = clean["lr"]
    _require(
        isinstance(lr, (int, float)) and not isinstance(lr, bool) and lr > 0,
        f"'training.lr' debe ser > 0, se recibió {lr!r}.",
    )
    clean["lr"] = float(lr)
    val_fraction = clean["val_fraction"]
    _require(
        isinstance(val_fraction, (int, float))
        and not isinstance(val_fraction, bool)
        and 0.0 <= float(val_fraction) < 1.0,
        f"'training.val_fraction' debe estar en [0, 1), se recibió {val_fraction!r}.",
    )
    clean["val_fraction"] = float(val_fraction)
    return clean


def validate_dataset(nn: str, dataset_cfg: Any) -> dict[str, Any]:
    _require(isinstance(dataset_cfg, dict), "'dataset' debe ser un objeto JSON con 'name'.")
    name = dataset_cfg.get("name")
    try:
        info = dataset_info(name)
    except KeyError as exc:
        raise ValidationError(str(exc).strip("'")) from exc
    _require(
        nn in info["nns"],
        f"el dataset '{name}' no es compatible con la NN '{nn}'. "
        f"Compatible con: {info['nns']}.",
    )
    params = dataset_cfg.get("params") or {}
    _require(isinstance(params, dict), "'dataset.params' debe ser un objeto JSON.")
    unknown = sorted(set(params) - set(info["defaults"]))
    if unknown and info["is_custom"]:
        raise ValidationError(
            f"el dataset custom '{name}' no acepta parámetros {unknown}: sus muestras ya "
            "están fijadas por índice. Crea otro dataset si necesitas otras."
        )
    _require(
        not unknown,
        f"el dataset '{name}' no acepta el/los parámetro(s) {unknown}. "
        f"Acepta: {sorted(info['defaults'])}.",
    )
    merged = effective_params(name, params)
    limit = merged.get("limit")
    if limit is not None:
        _positive_int(limit, "dataset.params.limit")
    return {"name": name, "params": merged}


def validate_train_config(
    config: dict[str, Any],
    *,
    source_experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Valida y normaliza la config de un entrenamiento.

    Con `init_from`, la arquitectura (`model`) la dicta el experimento origen: se
    reemplaza aquí para que los pesos siempre carguen (regla 14).
    """
    _require(isinstance(config, dict), "la config debe ser un objeto JSON.")
    nn = config.get("nn")
    _require(nn in NNS, f"NN '{nn}' desconocida. Disponibles: {sorted(NNS)}.")

    dataset = validate_dataset(nn, config.get("dataset"))
    init_from = config.get("init_from") or None

    if init_from is not None:
        _require(
            source_experiment is not None,
            f"el experimento origen '{init_from}' (init_from) no existe.",
        )
        source_cfg = source_experiment["config"]
        _require(
            source_cfg["nn"] == nn,
            f"init_from apunta al experimento '{init_from}', que es de la NN "
            f"'{source_cfg['nn']}' y no de '{nn}'. Solo se puede re-entrenar sobre "
            "un experimento de la misma NN.",
        )
        _require(
            source_experiment.get("has_checkpoint", False),
            f"el experimento '{init_from}' no tiene checkpoint 'best.pt' "
            "(¿se detuvo o falló antes de completar una época?). No se puede "
            "re-entrenar desde él.",
        )
        model = copy.deepcopy(source_cfg["model"])
    else:
        model = validate_model(
            config.get("model") or {},
            input_size=28,
            num_classes=10,
        )

    seed = _positive_int(config.get("seed", 1234), "seed", minimum=0)
    return {
        "nn": nn,
        "dataset": dataset,
        "model": model,
        "training": validate_training(config.get("training") or {}),
        "seed": seed,
        "init_from": init_from,
    }


# --- evaluaciones ----------------------------------------------------------


def validate_eval_config(
    config: dict[str, Any], *, experiment: dict[str, Any] | None
) -> dict[str, Any]:
    """Valida una evaluación ANTES de crearla: NN entrenada + dataset compatible."""
    _require(isinstance(config, dict), "la config de la evaluación debe ser un objeto JSON.")
    exp_id = config.get("experiment")
    _require(
        experiment is not None,
        f"el experimento '{exp_id}' no existe: no hay ninguna NN que probar.",
    )
    _require(
        experiment.get("has_checkpoint", False),
        f"el experimento '{exp_id}' no tiene checkpoint 'best.pt': entrénalo (al menos "
        "una época completa) antes de probarlo.",
    )
    nn = experiment["config"]["nn"]

    name = config.get("dataset") or experiment["config"]["dataset"]["name"]
    try:
        info = dataset_info(name)
    except KeyError as exc:
        raise ValidationError(str(exc).strip("'")) from exc
    _require(
        nn in info["nns"],
        f"el dataset '{name}' no es compatible con la NN '{nn}' del experimento "
        f"'{exp_id}'. Compatible con: {info['nns']}.",
    )

    split = config.get("split", "test")
    _require(
        split in ("train", "test"),
        f"split '{split}' inválido: debe ser 'train' o 'test'.",
    )

    limit = config.get("limit")
    if limit is not None:
        limit = _positive_int(limit, "limit")

    threshold = config.get("ambiguous_threshold", DEFAULT_AMBIGUOUS_THRESHOLD)
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and 0.0 < float(threshold) <= 1.0,
        f"'ambiguous_threshold' debe estar en (0, 1], se recibió {threshold!r}. "
        "Es el margen (p1 − p2) por debajo del cual una predicción se considera ambigua.",
    )

    return {
        "experiment": exp_id,
        "nn": nn,
        "dataset": name,
        "dataset_is_custom": info["is_custom"],
        "split": split,
        "limit": limit,
        "ambiguous_threshold": float(threshold),
    }


def validate_filter(raw: Any, *, num_classes: int = 10) -> dict[str, Any]:
    """Filtro de resultados: por resultado (acierto/fallo/ambigua) y por clases."""
    raw = raw or {}
    _require(isinstance(raw, dict), "'filter' debe ser un objeto JSON.")

    outcome = raw.get("outcome", "all")
    _require(
        outcome in OUTCOMES,
        f"'outcome' inválido: {outcome!r}. Debe ser uno de {list(OUTCOMES)}.",
    )

    clean: dict[str, Any] = {"outcome": outcome, "label": None, "pred": None}
    for field in ("label", "pred"):
        value = raw.get(field)
        if value is None or value == "":
            continue
        value = _positive_int(
            int(value) if isinstance(value, str) and value.isdigit() else value,
            field,
            minimum=0,
            maximum=num_classes - 1,
        )
        clean[field] = value
    return clean
