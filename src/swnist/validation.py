"""Validación de compatibilidad NN↔dataset↔checkpoint, con razones claras.

Toda restricción se comunica con un mensaje que explica el porqué y cómo
corregirla (feedback para el usuario). Se valida ANTES de crear el experimento
o la evaluación, para que la API responda 400 con la razón en vez de dejar un
experimento fallido.
"""

from swnist.data import registry as data_registry
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import NNS


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
        ws_model = _model_window(config)
        ws_data = data_registry.effective_window_size(
            dataset_cfg["name"], dataset_cfg.get("params", {}) or {})
        if ws_model is not None and ws_model != ws_data:
            raise ValueError(
                f"Medidas incompatibles: el dimensionador está configurado para "
                f"ventanas de {ws_model}×{ws_model} (model.window_size) pero el "
                f"dataset {dataset_cfg['name']!r} produce entradas de "
                f"{ws_data}×{ws_data}. Ajusta model.window_size a {ws_data} o "
                f"elige un dataset con ese tamaño de ventana. (El secuenciador "
                f"reutiliza el window_size del dimensionador, por eso debe "
                f"reflejar lo realmente entrenado.)")

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
        if init_from:
            init_dim = init_exp["config"].get("dimensionador_experiment")
            init_dim_exp = _get_experiment(registry, init_dim, "Re-entrenamiento (init_from)")
            fd_init = init_dim_exp["config"]["model"]["feature_dim"]
            fd_new = dim_exp["config"]["model"]["feature_dim"]
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
                f"de ventana {ws_model} (p. ej. mnist_windows con "
                f"window_size={ws_model}).")

    if nn == "secuenciador":
        dim_id = exp["config"].get("dimensionador_experiment")
        dim_exp = _get_experiment(
            registry, dim_id,
            f"Evaluación: el secuenciador {exp_id!r} referencia al dimensionador")
        _require_checkpoint(registry, dim_id, "Evaluación (dimensionador)")

    split = config.get("split", "test")
    if split not in ("train", "test"):
        raise ValueError(f"Split inválido: {split!r} (usa 'train' o 'test').")
    limit = config.get("limit")
    if limit is not None and int(limit) <= 0:
        raise ValueError("'limit' debe ser un entero positivo (o vacío para todo el set).")
    return {**config, "split": split}
