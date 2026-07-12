"""Dataset sintético de rectas/curvas (continuas/entrecortadas) para el dimensionador.

Cada muestra es una ventana cuadrada (default 5×5) con un trazo dibujado dentro y
un vector de descriptores geométricos como target (ver ``swnist.descriptors``),
conocido por construcción: no se "mide" del bitmap, se deriva de los parámetros con
los que se generó el trazo.

Todo el "set" se define por parámetros JSON-editables desde la UI (rangos de ángulo,
longitud, curvatura, patrón de guiones, grosor, y cuántas muestras): con solo
cambiar esos rangos se genera un conjunto distinto. Es determinista dado
(seed, train, idx): la RNG de cada muestra se deriva de su índice, así que un
experimento es reproducible.

Nunca se genera una ventana completamente vacía (siempre hay al menos un punto de
tinta), pero sí trazos cortos (``length_min`` pequeño).
"""

import numpy as np
import torch

from swnist.descriptors import IDX, NUM_DESCRIPTORS, category_index

IMAGE_SIZE = 28  # (no se usa aquí, pero mantiene la convención del paquete)

# Estadísticas de normalización de MNIST: el trazo se entrega normalizado para
# vivir en el MISMO dominio que las ventanas reales que verá el secuenciador.
_MEAN, _STD = 0.1307, 0.3081

# Parámetros (con default) que definen el conjunto de muestras. Son los "rangos de
# datos" JSON-editables desde la UI. `type` y `bounds` los usa la validación.
#   fraction: prob. de ese tipo de trazo   |   deg: ángulo en grados
#   length/curvature/dash: unidades de píxel de la ventana (salvo *_duty en [0,1])
STROKE_DEFAULTS: dict = {
    "window_size": 5,
    "num_samples": 20000,
    "curved_fraction": 0.5,    # P(trazo curvo); el resto son rectas
    "broken_fraction": 0.5,    # P(trazo entrecortado); el resto son continuos
    "angle_min_deg": 0.0,
    "angle_max_deg": 180.0,
    "length_min": 1.5,         # trazos cortos permitidos (nunca 0 → nunca vacío)
    "length_max": 6.0,
    "curvature_min": 0.15,     # sagitta/(longitud) de la curva; straightness = 1 - curvature
    "curvature_max": 0.6,
    "dash_duty_min": 0.35,     # fracción de tinta a lo largo del trazo entrecortado
    "dash_duty_max": 0.7,
    "dash_period_min": 1.2,    # periodo del patrón de guiones (px)
    "dash_period_max": 2.5,
    "stroke_width": 1.0,       # grosor del trazo (px)
    "empty_fraction": 0.1,     # fracción de ventanas VACÍAS (ink=0); su geometría
                               # se enmascara de la pérdida (ver descriptors.GEOMETRY)
}

# Validación: (tipo, mínimo, máximo) por parámetro. Los *_min/*_max además deben
# cumplir min ≤ max (se chequea aparte).
_SPEC: dict = {
    "window_size": ("int", 2, 28),
    "num_samples": ("int", 1, 10_000_000),
    "curved_fraction": ("float", 0.0, 1.0),
    "broken_fraction": ("float", 0.0, 1.0),
    "angle_min_deg": ("float", 0.0, 180.0),
    "angle_max_deg": ("float", 0.0, 180.0),
    "length_min": ("float", 0.5, 28.0),
    "length_max": ("float", 0.5, 28.0),
    "curvature_min": ("float", 0.0, 2.0),
    "curvature_max": ("float", 0.0, 2.0),
    "dash_duty_min": ("float", 0.05, 1.0),
    "dash_duty_max": ("float", 0.05, 1.0),
    "dash_period_min": ("float", 0.3, 28.0),
    "dash_period_max": ("float", 0.3, 28.0),
    "stroke_width": ("float", 0.3, 10.0),
    "empty_fraction": ("float", 0.0, 0.9),
}

_RANGE_PAIRS = [
    ("angle_min_deg", "angle_max_deg"),
    ("length_min", "length_max"),
    ("curvature_min", "curvature_max"),
    ("dash_duty_min", "dash_duty_max"),
    ("dash_period_min", "dash_period_max"),
]


def validate_params(params: dict) -> None:
    """Rechaza parámetros del dataset de trazos con valores inválidos (mensaje claro)."""
    for key, val in params.items():
        if key not in _SPEC:
            continue  # los desconocidos los caza la validación genérica del registro
        kind, lo, hi = _SPEC[key]
        if isinstance(val, bool):
            raise ValueError(f"{key} debe ser un número; recibido: {val!r}.")
        if kind == "int" and not isinstance(val, int):
            raise ValueError(f"{key} debe ser un entero; recibido: {val!r}.")
        if not isinstance(val, (int, float)):
            raise ValueError(f"{key} debe ser un número; recibido: {val!r}.")
        if not (lo <= val <= hi):
            raise ValueError(f"{key} debe estar en [{lo}, {hi}]; recibido: {val!r}.")
    eff = {**STROKE_DEFAULTS, **params}
    for lo_key, hi_key in _RANGE_PAIRS:
        if eff[lo_key] > eff[hi_key]:
            raise ValueError(
                f"Rango inválido: {lo_key}={eff[lo_key]} no puede ser mayor que "
                f"{hi_key}={eff[hi_key]}.")


def _render(points: np.ndarray, ws: int, stroke_width: float) -> np.ndarray:
    """Gris en [0, 1] (ws, ws) de un trazo dado por puntos (N, 2) en coords (fila, col).

    Anti-alias por distancia al punto más cercano del trazo (misma idea que
    swnist.data.strokes): intensidad 1 dentro del medio-grosor y caída lineal de
    1 px en el borde.
    """
    rows, cols = np.mgrid[0:ws, 0:ws]
    # Distancia de cada píxel (fila, col) al punto más cercano del trazo.
    d2 = ((rows[..., None] - points[:, 0]) ** 2
          + (cols[..., None] - points[:, 1]) ** 2)
    dist = np.sqrt(d2.min(axis=-1))
    return np.clip(stroke_width / 2.0 + 0.5 - dist, 0.0, 1.0)


class SyntheticStrokes(torch.utils.data.Dataset):
    """Ventanas con un trazo sintético y descriptores geométricos como target.

    Compatible con: dimensionador. ``__getitem__`` -> (ventana (1,ws,ws)
    normalizada, target (6,) float32). ``display_item`` -> (ventana, categoría),
    donde la categoría (0..3) es la derivada de los descriptores para las
    miniaturas/filtros de la pestaña Probar.
    """

    def __init__(self, train: bool = True, seed: int = 0, **params):
        cfg = {**STROKE_DEFAULTS, **params}
        self.train = bool(train)
        self.seed = int(seed)
        self.window_size = int(cfg["window_size"])
        self.num_samples = int(cfg["num_samples"])
        self.cfg = cfg

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.RandomState:
        # Train/test disjuntos: el offset por split cambia la semilla base.
        base = (self.seed * 2 + (0 if self.train else 1)) * 1_000_003 + idx
        return np.random.RandomState(base % (2**32))

    def _sample(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gris (ws, ws) en [0,1], target (7,), mask (7,)) determinista para idx.

        `mask` marca qué canales se supervisan: en una ventana vacía solo se
        supervisa `ink` (los canales de geometría no tienen sentido sin trazo).
        """
        cfg = self.cfg
        ws = self.window_size
        rng = self._rng(idx)

        # Ventana vacía: solo ancla el canal de tinta (ink=0), geometría enmascarada.
        # Con empty_fraction=0 no se consume RNG (muestras antiguas idénticas).
        if cfg["empty_fraction"] > 0 and rng.rand() < cfg["empty_fraction"]:
            gray = np.zeros((ws, ws), dtype=np.float32)
            target = np.zeros(NUM_DESCRIPTORS, dtype=np.float32)  # ink=0 incluido
            mask = np.zeros(NUM_DESCRIPTORS, dtype=np.float32)
            mask[IDX["ink"]] = 1.0
            return gray, target, mask

        curved = rng.rand() < cfg["curved_fraction"]
        broken = rng.rand() < cfg["broken_fraction"]
        theta = np.deg2rad(rng.uniform(cfg["angle_min_deg"], cfg["angle_max_deg"]))
        length = float(rng.uniform(cfg["length_min"], cfg["length_max"]))
        # dir en coords (fila=y, col=x): col ~ cos θ, fila ~ sin θ.
        dcol, drow = np.cos(theta), np.sin(theta)

        # Centro con jitter dentro de la ventana (los extremos pueden salirse un
        # poco → trazo recortado, pero nunca vacío).
        margin = 0.5
        cy = rng.uniform(margin, ws - 1 - margin)
        cx = rng.uniform(margin, ws - 1 - margin)
        half = length / 2.0
        p0 = np.array([cy - half * drow, cx - half * dcol])
        p1 = np.array([cy + half * drow, cx + half * dcol])

        curvature = 0.0
        if curved:
            curvature = float(rng.uniform(cfg["curvature_min"], cfg["curvature_max"]))
        # Puntos densos a lo largo del camino (recta o Bézier cuadrática).
        n = max(8, int(np.ceil(length * 4)) + 1)
        t = np.linspace(0.0, 1.0, n)
        if curvature > 0:
            # Control desplazado perpendicular a la cuerda por la sagitta.
            perp = np.array([-dcol, drow])  # perpendicular a dir
            sag = curvature * length
            sign = 1.0 if rng.rand() < 0.5 else -1.0
            mid = (p0 + p1) / 2.0 + sign * sag * perp
            tt = t[:, None]
            path = ((1 - tt) ** 2) * p0 + 2 * (1 - tt) * tt * mid + (tt ** 2) * p1
        else:
            path = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]

        # Continuidad: patrón de guiones a lo largo del parámetro de arco.
        duty = 1.0
        if broken:
            duty = float(rng.uniform(cfg["dash_duty_min"], cfg["dash_duty_max"]))
            period = float(rng.uniform(cfg["dash_period_min"], cfg["dash_period_max"]))
            # arco acumulado (aprox por la longitud entre muestras del path)
            seg = np.sqrt(((path[1:] - path[:-1]) ** 2).sum(axis=1))
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            phase = rng.uniform(0.0, period)
            inked = ((arc + phase) % period) < (duty * period)
            if not inked.any():
                inked[len(inked) // 2] = True  # nunca vacío
            path = path[inked]

        gray = _render(path, ws, float(cfg["stroke_width"]))

        target = np.zeros(NUM_DESCRIPTORS, dtype=np.float32)
        target[IDX["straightness"]] = 1.0 - curvature
        target[IDX["horizontal"]] = abs(np.cos(theta))
        target[IDX["vertical"]] = abs(np.sin(theta))
        target[IDX["angle_sin2"]] = np.sin(2 * theta)
        target[IDX["angle_cos2"]] = np.cos(2 * theta)
        target[IDX["continuity"]] = duty
        target[IDX["ink"]] = float(np.clip(gray.mean(), 0.0, 1.0))  # cobertura
        return gray, target, np.ones(NUM_DESCRIPTORS, dtype=np.float32)

    def __getitem__(self, idx: int):
        gray, target, mask = self._sample(idx)
        window = ((gray - _MEAN) / _STD).astype(np.float32)
        return (torch.from_numpy(window).unsqueeze(0),
                torch.from_numpy(target),
                torch.from_numpy(mask))

    def display_item(self, idx: int):
        """(ventana mostrable, índice de categoría) para miniaturas/filtros."""
        window, target, _ = self[idx]
        return window, category_index(target)
