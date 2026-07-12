"""Catálogo de datasets (builtin + custom) y su compatibilidad con cada NN.

La web app usa `list_datasets(nn)` para mostrar solo los datasets compatibles
con la NN seleccionada. Los datasets custom (subconjuntos filtrados, ver
`swnist.data.custom`) heredan la compatibilidad de su dataset base.
"""

from .custom import CustomDatasetStore, CustomSubset
from .datasets import (IMAGE_SIZE, MnistContourSequences, MnistFull,
                       MnistSlidingSequences, MnistWindows)

DATASETS = {
    "mnist_full": {
        "description": "Imágenes MNIST completas 28×28; con window_size < 28 entrena "
                       "con ventanas de cada imagen: sampling='random' toma "
                       "windows_per_image ventanas aleatorias, sampling='raster' recorre "
                       "la grilla window_size+stride completa (ignora windows_per_image). "
                       "Con empty_fraction > 0 esa fracción son ventanas vacías con la "
                       "clase extra 'no hay nada' (10). stroke_width > 0 uniformiza el "
                       "grosor del trazo a ~N px (0 = original).",
        "compatible_with": ["dimensionador"],
        "builder": MnistFull,
        "defaults": {"window_size": 5, "windows_per_image": 30, "sampling": "raster",
                     "stride": 5, "empty_fraction": 0.1, "stroke_width": 1},
        "full_image": True,   # la muestra visible es una imagen completa 28×28
    },
    "mnist_windows": {
        "description": "Ventanas de MNIST etiquetadas con el dígito de origen: "
                       "sampling='random' toma windows_per_image ventanas aleatorias por "
                       "imagen, sampling='raster' recorre la grilla window_size+stride "
                       "completa (ignora windows_per_image). Con empty_fraction > 0 esa "
                       "fracción son ventanas vacías con la clase extra 'no hay nada' "
                       "(10). stroke_width > 0 uniformiza el grosor del trazo a ~N px "
                       "(0 = original).",
        "compatible_with": ["dimensionador"],
        "builder": MnistWindows,
        "defaults": {"window_size": 5, "windows_per_image": 4, "sampling": "raster",
                     "stride": 5, "empty_fraction": 0.1, "stroke_width": 1},
        "full_image": False,
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
    samp = params.get("sampling")
    if samp is not None and samp not in ("random", "raster"):
        raise ValueError(
            f"sampling debe ser 'random' (windows_per_image ventanas aleatorias "
            f"por imagen) o 'raster' (la grilla determinista window_size+stride "
            f"que cubre la imagen, ignorando windows_per_image); recibido: {samp!r}.")
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
    ef = params.get("empty_fraction")
    if ef is not None and (isinstance(ef, bool) or not isinstance(ef, (int, float))
                           or not 0 <= ef < 1):
        raise ValueError(
            f"empty_fraction debe ser un número en [0, 1): la fracción de muestras "
            f"que son ventanas vacías con la clase 'no hay nada' (con 1 no quedaría "
            f"ningún dígito que aprender); recibido: {ef!r}.")
    if name not in DATASETS:
        for key in ("windows_per_image", "empty_fraction"):
            if key in params and params[key] != base_params.get(key):
                raise ValueError(
                    f"No se puede cambiar {key} en el dataset custom {name!r}: sus "
                    f"índices se guardaron con {key}={base_params.get(key)} y dejarían "
                    f"de apuntar a las mismas muestras (con otro empty_fraction hasta "
                    f"la etiqueta de una muestra cambiaría). Crea un subconjunto nuevo "
                    f"si necesitas otro valor.")
        if "sampling" in accepted:
            eff_sampling = base_params.get("sampling", "random")
            if "sampling" in params and params["sampling"] != eff_sampling:
                raise ValueError(
                    f"No se puede cambiar sampling en el dataset custom {name!r}: sus "
                    f"índices se guardaron con sampling={eff_sampling!r} y con otro "
                    f"muestreo dejarían de apuntar a las mismas ventanas. Crea un "
                    f"subconjunto nuevo si necesitas otro muestreo.")
            if eff_sampling == "raster":
                for key in ("window_size", "stride"):
                    if key in params and params[key] != base_params.get(key):
                        raise ValueError(
                            f"No se puede cambiar {key} en el dataset custom {name!r}: "
                            f"con sampling='raster' la grilla (window_size, stride) "
                            f"define cuántas ventanas salen de cada imagen, y sus "
                            f"índices se guardaron con la grilla window_size="
                            f"{base_params.get('window_size')}, stride="
                            f"{base_params.get('stride')}. Crea un subconjunto nuevo "
                            f"si necesitas otra grilla.")


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
