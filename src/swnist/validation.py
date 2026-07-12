"""Validación de compatibilidad NN↔dataset↔checkpoint, con razones claras.

Toda restricción se comunica con un mensaje que explica el porqué y cómo
corregirla (feedback para el usuario). Se valida ANTES de crear el experimento
o la evaluación, para que la API responda 400 con la razón en vez de dejar un
experimento fallido.
"""

import torch

from swnist.data import registry as data_registry
from swnist.data.datasets import IMAGE_SIZE
from swnist.descriptors import NAMES, NUM_DESCRIPTORS
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.secuenciador import POSITION_ENCODING
from swnist.nn_registry import NNS

# Un secuenciador anterior al cambio a posición relativa (2026-07-12) recibía la
# posición ABSOLUTA (x, y): mismos shapes, entrada con otro significado, así que
# sus pesos no son reutilizables. Se detecta por la ausencia de
# model.position_encoding en su config (las nuevas lo registran siempre).
_LEGACY_POSITIONS = (
    "se entrenó con la posición absoluta (x, y) como entrada, que ya no se usa: "
    "ahora la red recibe el desplazamiento (dx, dy) respecto al paso anterior. Sus "
    "pesos tienen el shape correcto pero la entrada significa otra cosa, así que no "
    "se pueden reutilizar. Entrena un secuenciador nuevo (su dimensionador se "
    "reutiliza tal cual).")


def _uses_relative_positions(exp_config: dict) -> bool:
    return (exp_config.get("model") or {}).get("position_encoding") == POSITION_ENCODING


def _get_experiment(registry: ExperimentRegistry, exp_id: str, what: str) -> dict:
    try:
        return registry.get_experiment(exp_id)
    except FileNotFoundError:
        raise ValueError(f"{what}: el experimento {exp_id!r} no existe "
                         f"(¿fue eliminado?).")


def _require_checkpoint(registry: ExperimentRegistry, exp_id: str, what: str) -> None:
    if not (registry.checkpoints_dir(exp_id) / "best.pt").exists():
        raise ValueError(
            f"{what}: el experimento {exp_id!r} no tiene checkpoint 'best.pt' "
            f"(quizá falló o se detuvo antes de la primera época). Elige un "
            f"experimento completado.")


def _check_dataset_for_nn(nn: str, dataset_cfg: dict) -> dict:
    name = dataset_cfg.get("name")
    params = dataset_cfg.get("params", {}) or {}
    try:
        info = data_registry.dataset_info(name)
    except KeyError:
        raise ValueError(f"Dataset desconocido: {name!r}.")
    if nn not in info["compatible_with"]:
        raise ValueError(
            f"El dataset {name!r} no es compatible con la NN {nn!r}: produce "
            f"muestras para {info['compatible_with']}. Elige un dataset "
            f"compatible en el desplegable.")
    data_registry.validate_dataset_params(name, params)
    return info


def _model_window(config: dict) -> int | None:
    model = config.get("model") or {}
    ws = model.get("window_size")
    return int(ws) if ws is not None else None


# Parámetros que definen la trayectoria de un dataset de secuencias: stride en
# el recorrido raster, num_steps en el recorrido por el trazo (contour).
TRAJECTORY_KEYS = ("stride", "num_steps")


def dimensionador_feature_dim(registry: ExperimentRegistry, exp_id: str) -> int:
    """Tamaño del vector de features (descriptores) de un dimensionador entrenado.

    `feature_dim` lo fija el contrato de descriptores, así que las configs nuevas lo
    registran pero las anteriores no lo tienen (bug del 2026-07-12: leerlo a pelo de
    config.model reventaba con KeyError → 500 al re-entrenar un secuenciador). Se
    resuelve en orden: config del experimento → checkpoint → contrato actual.
    """
    exp = registry.get_experiment(exp_id)
    fd = (exp["config"].get("model") or {}).get("feature_dim")
    if fd is not None:
        return int(fd)
    ckpt = registry.checkpoints_dir(exp_id) / "best.pt"
    if ckpt.exists():
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            fd = (data.get("model_config") or {}).get("feature_dim")
            if fd is not None:
                return int(fd)
        except Exception:  # checkpoint ilegible: cae al contrato actual
            pass
    return NUM_DESCRIPTORS


def sequence_effective_params(exp_config: dict, registry: ExperimentRegistry) -> dict:
    """Ventana y trayectoria con las que realmente se entrenó un secuenciador.

    La trayectoria es ENTRADA de la red (la posición por paso, de la que sale el
    desplazamiento (dx, dy) que consume el secuenciador): cambiarla en inferencia
    evalúa fuera de distribución (bug del 2026-07-11: entrenar con stride 5 —36
    pasos— y evaluar con stride 7 —25 pasos— hundía la accuracy de 0.91 a 0.56).
    window_size lo dicta su dimensionador; stride/num_steps son los efectivos del
    dataset de entrenamiento (defaults ⊕ params).
    """
    dim_exp = _get_experiment(registry, exp_config["dimensionador_experiment"],
                              "Secuenciador: dimensionador")
    out = {"window_size": int(dim_exp["config"]["model"]["window_size"])}
    ds = exp_config.get("dataset") or {}
    eff = data_registry.effective_params(ds.get("name"), ds.get("params") or {})
    for key in TRAJECTORY_KEYS:
        if eff.get(key) is not None:
            out[key] = int(eff[key])
    return out


def validate_train_config(nn: str, config: dict, registry: ExperimentRegistry) -> dict:
    """Valida (y normaliza) una config de entrenamiento. Devuelve la config final.

    Si `init_from` está presente, el modelo se construye con la arquitectura del
    experimento origen (config.model se reemplaza) para que los pesos carguen.
    """
    if nn not in NNS:
        raise ValueError(f"NN desconocida: {nn!r}. Disponibles: {sorted(NNS)}.")
    config = dict(config)

    init_from = config.get("init_from")
    if init_from:
        init_exp = _get_experiment(registry, init_from, "Re-entrenamiento (init_from)")
        init_nn = init_exp["config"].get("nn")
        if init_nn != nn:
            raise ValueError(
                f"Re-entrenamiento (init_from): {init_from!r} es un experimento de "
                f"{init_nn!r}, pero estás entrenando un {nn!r}. Los pesos solo se "
                f"pueden continuar en la misma arquitectura.")
        _require_checkpoint(registry, init_from, "Re-entrenamiento (init_from)")
        # La arquitectura la dicta el checkpoint origen: si no coincidiera, los
        # pesos no cargarían (shapes distintos).
        config["model"] = dict(init_exp["config"]["model"])

    dataset_cfg = config.get("dataset") or {}
    _check_dataset_for_nn(nn, dataset_cfg)

    if nn == "dimensionador":
        channels = (config.get("model") or {}).get("channels")
        if channels is not None and (
                not isinstance(channels, (list, tuple)) or not channels
                or any(not isinstance(c, int) or c <= 0 for c in channels)):
            raise ValueError(
                f"model.channels debe ser una lista con al menos un entero positivo "
                f"(un canal por bloque convolucional, p. ej. [16, 32, 64]); "
                f"recibido: {channels!r}.")
        ws_model = _model_window(config)
        ws_data = data_registry.effective_window_size(
            dataset_cfg["name"], dataset_cfg.get("params", {}) or {})
        if ws_model is not None and ws_model != ws_data:
            raise ValueError(
                f"Medidas incompatibles: el dimensionador está configurado para "
                f"ventanas de {ws_model}×{ws_model} (model.window_size) pero el "
                f"dataset {dataset_cfg['name']!r} produce entradas de "
                f"{ws_data}×{ws_data}. Ajusta model.window_size a {ws_data} o "
                f"elige un window_size igual en el dataset. (El secuenciador "
                f"reutiliza el window_size del dimensionador, por eso debe "
                f"reflejar lo realmente entrenado.)")
        # feature_dim lo fija el contrato de descriptores: se registra en la config
        # (para que quede trazable y el secuenciador pueda leerlo sin abrir el
        # checkpoint). Con init_from, los pesos del origen solo cargan si se entrenó
        # con el mismo número de descriptores.
        if init_from:
            fd_origin = dimensionador_feature_dim(registry, init_from)
            if fd_origin != NUM_DESCRIPTORS:
                raise ValueError(
                    f"Re-entrenamiento incompatible: el dimensionador {init_from!r} se "
                    f"entrenó con {fd_origin} descriptores, pero el contrato actual "
                    f"tiene {NUM_DESCRIPTORS} ({', '.join(NAMES)}). Sus pesos no cargan "
                    f"en la arquitectura actual: entrena un dimensionador nuevo.")
        config["model"] = {**(config.get("model") or {}), "feature_dim": NUM_DESCRIPTORS}

    if nn == "secuenciador":
        dim_id = config.get("dimensionador_experiment")
        if not dim_id:
            raise ValueError(
                "Falta 'dimensionador_experiment': el secuenciador consume las "
                "features de un dimensionador ya entrenado. Entrena primero un "
                "dimensionador y selecciónalo en el formulario.")
        dim_exp = _get_experiment(registry, dim_id, "Dimensionador")
        if dim_exp["config"].get("nn") != "dimensionador":
            raise ValueError(
                f"{dim_id!r} no es un experimento de dimensionador.")
        _require_checkpoint(registry, dim_id, "Dimensionador")
        dim_ws = int(dim_exp["config"]["model"]["window_size"])
        if dim_ws >= IMAGE_SIZE:
            raise ValueError(
                f"El dimensionador {dim_id!r} se entrenó con ventana "
                f"{dim_ws}×{dim_ws} (la imagen completa): la ventana deslizante "
                f"no tendría dónde moverse y la secuencia colapsaría a un solo "
                f"paso, que es justo lo que el secuenciador debe evitar. Entrena "
                f"un dimensionador con model.window_size < {IMAGE_SIZE} (p. ej. "
                f"5 con synthetic_strokes y window_size=5) y selecciónalo aquí.")
        # La ventana de la secuencia la dicta el dimensionador, y la trayectoria
        # efectiva (stride o num_steps, dados o heredados de los defaults) se
        # fija en la config para que quede registrado el valor realmente usado:
        # evaluaciones y predict deben reproducir esta misma trayectoria.
        params = {**(dataset_cfg.get("params") or {}), "window_size": dim_ws}
        eff = data_registry.effective_params(dataset_cfg["name"], params)
        for key in TRAJECTORY_KEYS:
            if eff.get(key) is not None:
                params[key] = int(eff[key])
        dataset_cfg = {**dataset_cfg, "params": params}
        config["dataset"] = dataset_cfg
        # La codificación de la posición es parte del contrato de entrada de la red
        # (dx, dy relativos al paso anterior): se registra en la config para que
        # quede trazable y para distinguir los secuenciadores antiguos (absolutos).
        enc = (config.get("model") or {}).get("position_encoding", POSITION_ENCODING)
        if enc != POSITION_ENCODING:
            raise ValueError(
                f"model.position_encoding={enc!r} no existe: el secuenciador recibe "
                f"el desplazamiento (dx, dy) respecto al paso anterior "
                f"({POSITION_ENCODING!r}), no la posición absoluta. Quita el campo o "
                f"ponlo en {POSITION_ENCODING!r}.")
        config["model"] = {**(config.get("model") or {}),
                           "position_encoding": POSITION_ENCODING}
        if init_from:
            if not _uses_relative_positions(init_exp["config"]):
                raise ValueError(
                    f"Re-entrenamiento incompatible: el secuenciador {init_from!r} "
                    f"{_LEGACY_POSITIONS}")
            init_dim = init_exp["config"].get("dimensionador_experiment")
            _get_experiment(registry, init_dim, "Re-entrenamiento (init_from)")
            fd_init = dimensionador_feature_dim(registry, init_dim)
            fd_new = dimensionador_feature_dim(registry, dim_id)
            if fd_init != fd_new:
                raise ValueError(
                    f"Re-entrenamiento incompatible: el secuenciador {init_from!r} "
                    f"se entrenó con features de dimensión {fd_init} "
                    f"(dimensionador {init_dim!r}), pero el dimensionador "
                    f"seleccionado {dim_id!r} produce dimensión {fd_new}. Elige un "
                    f"dimensionador con feature_dim={fd_init} o entrena un "
                    f"secuenciador nuevo.")
    return config


def validate_eval_config(config: dict, registry: ExperimentRegistry) -> dict:
    """Valida una config de evaluación {experiment, dataset, split, limit}."""
    exp_id = config.get("experiment")
    if not exp_id:
        raise ValueError("Falta 'experiment': elige la NN (experimento) a evaluar.")
    exp = _get_experiment(registry, exp_id, "Evaluación")
    nn = exp["config"].get("nn")
    _require_checkpoint(registry, exp_id, "Evaluación")

    dataset_cfg = config.get("dataset") or {}
    _check_dataset_for_nn(nn, dataset_cfg)

    if nn == "dimensionador":
        ws_model = exp["config"]["model"].get("window_size")
        ws_data = data_registry.effective_window_size(
            dataset_cfg["name"], dataset_cfg.get("params", {}) or {})
        if ws_model is not None and int(ws_model) != ws_data:
            raise ValueError(
                f"Medidas incompatibles: {exp_id!r} se entrenó con ventanas de "
                f"{ws_model}×{ws_model}, pero el dataset {dataset_cfg['name']!r} "
                f"produce entradas de {ws_data}×{ws_data}. Evalúalo con un dataset "
                f"de ventana {ws_model} (p. ej. synthetic_strokes con "
                f"window_size={ws_model}).")

    if nn == "secuenciador":
        if not _uses_relative_positions(exp["config"]):
            raise ValueError(
                f"El secuenciador {exp_id!r} {_LEGACY_POSITIONS}")
        dim_id = exp["config"].get("dimensionador_experiment")
        _get_experiment(
            registry, dim_id,
            f"Evaluación: el secuenciador {exp_id!r} referencia al dimensionador")
        _require_checkpoint(registry, dim_id, "Evaluación (dimensionador)")
        # La trayectoria (ventana + stride/num_steps) la dicta el entrenamiento:
        # se fija en la config de la evaluación (valores efectivos, no los del
        # formulario) y pedir otra explícitamente es un 400 con la razón.
        eff = sequence_effective_params(exp["config"], registry)
        params = dict(dataset_cfg.get("params") or {})
        # El dataset elegido debe usar el mismo TIPO de recorrido que el del
        # entrenamiento (raster con stride vs. trazo con num_steps): si no
        # acepta el parámetro de trayectoria aprendido, la trayectoria sería
        # otra y la evaluación fuera de distribución.
        accepted = data_registry.effective_params(dataset_cfg["name"], {})
        train_ds = (exp["config"].get("dataset") or {}).get("name")
        for key in TRAJECTORY_KEYS:
            if key in eff and key not in accepted:
                raise ValueError(
                    f"El secuenciador {exp_id!r} se entrenó con el recorrido de "
                    f"{train_ds!r} ({key}={eff[key]}), pero {dataset_cfg['name']!r} "
                    f"usa otro tipo de recorrido (no acepta {key}). La trayectoria "
                    f"es entrada de la red: evalúalo con un dataset del mismo tipo "
                    f"de recorrido (p. ej. {train_ds!r} o un custom basado en él).")
        reasons = {
            "window_size": "es la ventana de su dimensionador",
            "stride": "define la trayectoria que la red aprendió (el desplazamiento "
                      "(dx, dy) por paso es entrada de la red); con otro stride la "
                      "evaluación sería fuera de distribución y engañosa",
            "num_steps": "define la trayectoria que la red aprendió (el número de "
                         "pasos del recorrido por el trazo es entrada de la red); "
                         "con otro num_steps la evaluación sería fuera de "
                         "distribución y engañosa",
        }
        for key, val in eff.items():
            given = params.get(key)
            if given is not None and int(given) != val:
                raise ValueError(
                    f"El secuenciador {exp_id!r} se entrenó con {key}={val}, que "
                    f"{reasons[key]}. No se puede evaluar con {key}={given}: usa "
                    f"{key}={val} (el formulario lo sincroniza solo) o entrena/"
                    f"re-entrena un secuenciador con {key}={given}.")
            params[key] = val
        config["dataset"] = {**dataset_cfg, "params": params}

    split = config.get("split", "test")
    if split not in ("train", "test"):
        raise ValueError(f"Split inválido: {split!r} (usa 'train' o 'test').")
    limit = config.get("limit")
    if limit is not None and int(limit) <= 0:
        raise ValueError("'limit' debe ser un entero positivo (o vacío para todo el set).")
    return {**config, "split": split}
