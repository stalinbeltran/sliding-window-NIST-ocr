"""Catálogo de datasets (builtin + custom) y su compatibilidad con cada NN.

La web app usa `list_datasets(nn)` para mostrar solo los datasets compatibles
con la NN seleccionada. Los datasets custom (subconjuntos filtrados, ver
`swnist.data.custom`) heredan la compatibilidad de su dataset base.
"""

from .custom import CustomDatasetStore, CustomSubset
from .datasets import IMAGE_SIZE, MnistFull, MnistSlidingSequences, MnistWindows

DATASETS = {
    "mnist_full": {
        "description": "Imágenes MNIST completas 28×28; con window_size < 28 entrena "
                       "con ventanas aleatorias de cada imagen (windows_per_image por imagen).",
        "compatible_with": ["dimensionador"],
        "builder": MnistFull,
        "defaults": {"window_size": 28, "windows_per_image": 1},
        "full_image": True,   # la muestra visible es una imagen completa 28×28
    },
    "mnist_windows": {
        "description": "Ventanas aleatorias de MNIST etiquetadas con el dígito de origen.",
        "compatible_with": ["dimensionador"],
        "builder": MnistWindows,
        "defaults": {"window_size": 14, "windows_per_image": 4},
        "full_image": False,
    },
    "mnist_sliding_sequences": {
        "description": "Secuencias de ventanas (recorrido raster) con posición (x, y) por paso.",
        "compatible_with": ["secuenciador"],
        "builder": MnistSlidingSequences,
        "defaults": {"window_size": 14, "stride": 7},
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
    else:
        d = custom_store.get(name)
        accepted = DATASETS[d["base"]["name"]]["defaults"]
        base_params = {**accepted, **d["base"]["params"]}
    unknown = set(params) - set(accepted)
    if unknown:
        raise ValueError(
            f"Parámetros no válidos para el dataset {name!r}: {sorted(unknown)}. "
            f"Parámetros aceptados: {sorted(accepted) or ['(ninguno)']}. Revisa que "
            f"config.dataset.params corresponda al dataset seleccionado."
        )
    ws = params.get("window_size")
    if ws is not None and not (isinstance(ws, int) and 1 <= ws <= IMAGE_SIZE):
        raise ValueError(
            f"window_size debe ser un entero entre 1 y {IMAGE_SIZE} (las imágenes "
            f"MNIST son de {IMAGE_SIZE}×{IMAGE_SIZE}); recibido: {ws!r}.")
    for key in ("windows_per_image", "stride"):
        v = params.get(key)
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"{key} debe ser un entero ≥ 1; recibido: {v!r}.")
    if name not in DATASETS and "windows_per_image" in params and \
            params["windows_per_image"] != base_params.get("windows_per_image"):
        raise ValueError(
            f"No se puede cambiar windows_per_image en el dataset custom {name!r}: "
            f"sus índices se guardaron con windows_per_image="
            f"{base_params.get('windows_per_image')} y dejarían de apuntar a las "
            f"mismas muestras. Crea un subconjunto nuevo si necesitas otro valor.")


def effective_window_size(name: str, params: dict) -> int:
    """Lado de la entrada 2D que producirá este dataset (para chequear compatibilidad)."""
    if name in DATASETS:
        merged = {**DATASETS[name]["defaults"], **params}
    else:
        d = custom_store.get(name)
        merged = {**DATASETS[d["base"]["name"]]["defaults"], **d["base"]["params"], **params}
    return int(merged.get("window_size", IMAGE_SIZE))


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
