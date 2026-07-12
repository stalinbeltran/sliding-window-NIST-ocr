"""Validación de compatibilidad NN↔dataset↔checkpoint, con razones claras.

Toda restricción se comunica con un mensaje que explica el porqué y cómo
corregirla (feedback para el usuario). Se valida ANTES de crear el experimento
o la evaluación, para que la API responda 400 con la razón en vez de dejar un
experimento fallido.
"""

from swnist.data import registry as data_registry
from swnist.data.datasets import IMAGE_SIZE
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


def sequence_effective_params(exp_config: dict, registry: ExperimentRegistry) -> dict:
    """Ventana y stride con los que realmente se entrenó un secuenciador.

    La trayectoria (posición (x, y) por paso) es ENTRADA de la red: cambiarla en
    inferencia evalúa fuera de distribución (bug del 2026-07-11: entrenar con
    stride 5 —36 pasos— y evaluar con stride 7 —25 pasos— hundía la accuracy de
    0.91 a 0.56). window_size lo dicta su dimensionador; stride es el efectivo
    del dataset de entrenamiento (defaults ⊕ params).
    """
    dim_exp = _get_experiment(registry, exp_config["dimensionador_experiment"],
                              "Secuenciador: dimensionador")
    out = {"window_size": int(dim_exp["config"]["model"]["window_size"])}
    ds = exp_config.get("dataset") or {}
    eff = data_registry.effective_params(ds.get("name"), ds.get("params") or {})
    if eff.get("stride") is not None:
        out["stride"] = int(eff["stride"])
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
        dim_ws = int(dim_exp["config"]["model"]["window_size"])
        if dim_ws >= IMAGE_SIZE:
            raise ValueError(
                f"El dimensionador {dim_id!r} se entrenó con ventana "
                f"{dim_ws}×{dim_ws} (la imagen completa): la ventana deslizante "
                f"no tendría dónde moverse y la secuencia colapsaría a un solo "
                f"paso, que es justo lo que el secuenciador debe evitar. Entrena "
                f"un dimensionador con model.window_size < {IMAGE_SIZE} (p. ej. "
                f"14 con mnist_full y window_size=14) y selecciónalo aquí.")
        # La ventana de la secuencia la dicta el dimensionador, y el stride
        # efectivo (dado o heredado de los defaults) se fija en la config para
        # que quede registrado el valor realmente usado: evaluaciones y predict
        # deben reproducir esta misma trayectoria.
        params = {**(dataset_cfg.get("params") or {}), "window_size": dim_ws}
        eff = data_registry.effective_params(dataset_cfg["name"], params)
        if eff.get("stride") is not None:
            params["stride"] = int(eff["stride"])
        dataset_cfg = {**dataset_cfg, "params": params}
        config["dataset"] = dataset_cfg
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
        _get_experiment(
            registry, dim_id,
            f"Evaluación: el secuenciador {exp_id!r} referencia al dimensionador")
        _require_checkpoint(registry, dim_id, "Evaluación (dimensionador)")
        # La trayectoria (ventana + stride) la dicta el entrenamiento: se fija en
        # la config de la evaluación (valores efectivos, no los del formulario) y
        # pedir otra explícitamente es un 400 con la razón.
        eff = sequence_effective_params(exp["config"], registry)
        params = dict(dataset_cfg.get("params") or {})
        reasons = {
            "window_size": "es la ventana de su dimensionador",
            "stride": "define la trayectoria que la red aprendió (las posiciones "
                      "(x, y) por paso son entrada de la red); con otro stride la "
                      "evaluación sería fuera de distribución y engañosa",
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
