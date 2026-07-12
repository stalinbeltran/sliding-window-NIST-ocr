"""Catálogo de datasets (builtin + custom) y su compatibilidad con cada NN.

La web app usa `list_datasets(nn)` para mostrar solo los datasets compatibles
con la NN seleccionada. Los datasets custom (subconjuntos filtrados, ver
`swnist.data.custom`) heredan la compatibilidad de su dataset base.
"""

from .custom import CustomDatasetStore, CustomSubset
from .datasets import (IMAGE_SIZE, MnistContourSequences,
                       MnistSlidingSequences)
from .synthstrokes import STROKE_DEFAULTS, SyntheticStrokes
from .synthstrokes import validate_params as _validate_stroke_params

DATASETS = {
    "synthetic_strokes": {
        "description": "Dataset sintético de rectas/curvas (continuas/entrecortadas) "
                       "para el dimensionador: cada muestra es una ventana cuadrada "
                       "(default 5×5) con un trazo y un vector de descriptores "
                       "geométricos como target (rectitud, horizontal, vertical, "
                       "ángulo, continuidad). El conjunto se define por rangos "
                       "JSON-editables (ángulo, longitud, curvatura, guiones, grosor) "
                       "y num_samples. Nunca hay ventanas vacías; sí trazos cortos.",
        "compatible_with": ["dimensionador"],
        "builder": SyntheticStrokes,
        "defaults": dict(STROKE_DEFAULTS),
        "full_image": False,   # la muestra visible es la ventana misma
    },
    "mnist_sliding_sequences": {
        "description": "Secuencias de ventanas (recorrido raster) con posición (x, y) "
                       "por paso. stroke_width > 0 uniformiza el grosor del trazo a "
                       "~N px (0 = original).",
        "compatible_with": ["secuenciador"],
        "builder": MnistSlidingSequences,
        "defaults": {"window_size": 14, "stride": 7, "stroke_width": 1},
        "full_image": True,
    },
    "mnist_contour_sequences": {
        "description": "Secuencias de ventanas que siguen el trazo del carácter "
                       "(esqueleto → camino → num_steps posiciones) con posición "
                       "(x, y) por paso; la trayectoria depende de cada muestra. "
                       "stroke_width > 0 uniformiza el grosor del trazo a ~N px "
                       "(0 = original).",
        "compatible_with": ["secuenciador"],
        "builder": MnistContourSequences,
        "defaults": {"window_size": 14, "num_steps": 12, "stroke_width": 1},
        "full_image": True,
    },
}

# Almacén de datasets custom; los tests lo sustituyen por uno temporal.
custom_store = CustomDatasetStore()


def _custom_entry(summary: dict) -> dict:
    base_spec = DATASETS[summary["base"]["name"]]
    return {
        "name": summary["name"],
        "description": summary["description"] or
            f"Subconjunto de {summary['base']['name']} ({summary['count']} muestras).",
        "compatible_with": base_spec["compatible_with"],
        # Params efectivos del base: lo que el frontend muestra/edita al seleccionarlo.
        "defaults": {**base_spec["defaults"], **summary["base"]["params"]},
        "custom": True,
        "base": summary["base"],
        "count": summary["count"],
    }


def list_datasets(nn: str | None = None) -> list[dict]:
    out = []
    for name, spec in DATASETS.items():
        if nn is None or nn in spec["compatible_with"]:
            out.append({
                "name": name,
                "description": spec["description"],
                "compatible_with": spec["compatible_with"],
                "defaults": spec["defaults"],
                "custom": False,
            })
    for summary in custom_store.list():
        entry = _custom_entry(summary)
        if nn is None or nn in entry["compatible_with"]:
            out.append(entry)
    return out


def dataset_info(name: str) -> dict:
    """Entrada de catálogo (builtin o custom); KeyError si no existe."""
    if name in DATASETS:
        spec = DATASETS[name]
        return {"name": name, "description": spec["description"],
                "compatible_with": spec["compatible_with"], "defaults": spec["defaults"],
                "custom": False, "full_image": spec["full_image"]}
    summary = custom_store.get_summary(name)  # KeyError si no existe
    entry = _custom_entry(summary)
    entry["full_image"] = DATASETS[summary["base"]["name"]]["full_image"]
    return entry


def validate_dataset_params(name: str, params: dict) -> None:
    """Rechaza parámetros que el dataset no acepta o con valores inválidos, con mensaje claro."""
    if name in DATASETS:
        accepted = DATASETS[name]["defaults"]
        base_params = accepted
        base_name = name
    else:
        d = custom_store.get(name)
        base_name = d["base"]["name"]
        accepted = DATASETS[base_name]["defaults"]
        base_params = {**accepted, **d["base"]["params"]}
    unknown = set(params) - set(accepted)
    if unknown:
        raise ValueError(
            f"Parámetros no válidos para el dataset {name!r}: {sorted(unknown)}. "
            f"Parámetros aceptados: {sorted(accepted) or ['(ninguno)']}. Revisa que "
            f"config.dataset.params corresponda al dataset seleccionado."
        )

    if base_name == "synthetic_strokes":
        # El dataset de trazos valida sus propios rangos (ángulo, longitud,
        # curvatura, guiones, grosor, num_samples).
        _validate_stroke_params(params)
        if name not in DATASETS:
            # Todos sus params definen la identidad/target de cada muestra: cambiar
            # cualquiera remapearía los índices guardados del subconjunto.
            changed = [k for k, v in params.items() if base_params.get(k) != v]
            if changed:
                raise ValueError(
                    f"No se puede cambiar {sorted(changed)} en el dataset custom "
                    f"{name!r}: los parámetros de generación definen cada muestra y "
                    f"sus índices dejarían de apuntar a los mismos trazos. Crea un "
                    f"subconjunto nuevo si necesitas otros rangos.")
        return

    # Datasets de secuencias (secuenciador): ventana + trayectoria (stride/num_steps)
    # y grosor de trazo.
    ws = params.get("window_size")
    if ws is not None and not (isinstance(ws, int) and 1 <= ws <= IMAGE_SIZE):
        raise ValueError(
            f"window_size debe ser un entero entre 1 y {IMAGE_SIZE} (las imágenes "
            f"MNIST son de {IMAGE_SIZE}×{IMAGE_SIZE}); recibido: {ws!r}.")
    v = params.get("stride")
    if v is not None and (not isinstance(v, int) or v < 1):
        raise ValueError(f"stride debe ser un entero ≥ 1; recibido: {v!r}.")
    ns = params.get("num_steps")
    if ns is not None and (not isinstance(ns, int) or ns < 2):
        raise ValueError(
            f"num_steps debe ser un entero ≥ 2 (con un solo paso no hay secuencia "
            f"que recorrer); recibido: {ns!r}.")
    sw = params.get("stroke_width")
    if sw is not None and (isinstance(sw, bool) or not isinstance(sw, int)
                           or not 0 <= sw <= IMAGE_SIZE):
        raise ValueError(
            f"stroke_width debe ser un entero entre 0 y {IMAGE_SIZE}: 0 deja el "
            f"trazo original y N > 0 redibuja el carácter con trazo de ~N px de "
            f"grosor uniforme (esqueleto re-entintado); recibido: {sw!r}.")


def effective_params(name: str, params: dict) -> dict:
    """Params con los que realmente se construirá el dataset (defaults ⊕ base custom ⊕ params).

    Es la única fuente de verdad para window_size/stride efectivos: todo chequeo
    de compatibilidad debe mirar esto y no los params crudos de una config.
    """
    if name in DATASETS:
        return {**DATASETS[name]["defaults"], **params}
    d = custom_store.get(name)  # KeyError si no existe
    return {**DATASETS[d["base"]["name"]]["defaults"], **d["base"]["params"], **params}


def effective_window_size(name: str, params: dict) -> int:
    """Lado de la entrada 2D que producirá este dataset (para chequear compatibilidad)."""
    return int(effective_params(name, params).get("window_size", IMAGE_SIZE))


def build_dataset(name: str, params: dict, train: bool, seed: int):
    if name in DATASETS:
        validate_dataset_params(name, params)
        spec = DATASETS[name]
        kwargs = {**spec["defaults"], **params}
        return spec["builder"](train=train, seed=seed, **kwargs)

    try:
        d = custom_store.get(name)
    except KeyError:
        raise KeyError(f"Dataset desconocido: {name!r}")
    validate_dataset_params(name, params)
    base = d["base"]
    spec = DATASETS[base["name"]]
    kwargs = {**spec["defaults"], **base["params"], **params}
    if not train:
        # Test final: split de test completo del dataset base.
        return spec["builder"](train=False, seed=seed, **kwargs)
    base_ds = spec["builder"](train=(base["split"] == "train"), seed=base["seed"], **kwargs)
    return CustomSubset(base_ds, d["indices"])
