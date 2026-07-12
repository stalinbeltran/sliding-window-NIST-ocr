"""Contrato de los descriptores geométricos que produce el dimensionador.

El dimensionador dejó de clasificar dígitos: ahora **describe el trazo** que ve en
su ventana con estas salidas interpretables, que son a la vez
  (a) los *targets* con los que se entrena sobre el dataset sintético de rectas/curvas,
  (b) lo que ``forward()`` regresa (crudo) para la pérdida, y
  (c) lo que ``features()`` entrega al secuenciador (combinador).

Este módulo es la única fuente de verdad del conjunto y el orden de los canales:
lo importan el dataset (targets), el modelo (activaciones), el entrenamiento
(pérdida/métricas), el runner de evaluación y la inferencia/UI.

Ángulo: la orientación de un trazo es periódica (0° y 180° son la misma), así que
se codifica como el par (sin 2θ, cos 2θ) —sin salto en el borde— y se decodifica a
un escalar θ∈[0, π) solo para mostrarlo.
"""

import math

import torch

# (nombre, activación) por canal, en orden fijo. La activación fija el rango
# natural: sigmoid -> [0, 1]; tanh -> [-1, 1]. Del crudo (forward) al valor
# interpretable (features) se pasa aplicando estas activaciones (activate()).
DESCRIPTORS: list[tuple[str, str]] = [
    ("straightness", "sigmoid"),  # 1 = recta, 0 = muy curva
    ("horizontal", "sigmoid"),    # |cos θ| de la orientación
    ("vertical", "sigmoid"),      # |sin θ| de la orientación
    ("angle_sin2", "tanh"),       # sin(2θ)
    ("angle_cos2", "tanh"),       # cos(2θ)
    ("continuity", "sigmoid"),    # 1 = continua, 0 = entrecortada (duty de tinta)
    ("ink", "sigmoid"),           # cantidad de tinta (cobertura = gris medio); 0 = ventana vacía
]

# Canales de "geometría": solo tienen sentido cuando hay un trazo. En una ventana
# vacía se enmascaran de la pérdida (no se supervisan); `ink` siempre se supervisa.
GEOMETRY = ("straightness", "horizontal", "vertical", "angle_sin2", "angle_cos2",
            "continuity")

# Por debajo de esta tinta (gris medio) la ventana se considera "vacía" para la
# categoría discreta (ver category_index) y en la UI.
EMPTY_INK = 0.04

NAMES: list[str] = [n for n, _ in DESCRIPTORS]
NUM_DESCRIPTORS: int = len(DESCRIPTORS)
IDX: dict[str, int] = {n: i for i, (n, _) in enumerate(DESCRIPTORS)}


def activate(raw: torch.Tensor) -> torch.Tensor:
    """Crudo (..., 6) -> descriptores en rango natural (activación por canal).

    Conserva el grafo de autograd (se usa dentro de la pérdida): construye la
    salida con ``torch.stack`` en vez de asignaciones in-place.
    """
    cols = []
    for i, (_, act) in enumerate(DESCRIPTORS):
        c = raw[..., i]
        cols.append(torch.sigmoid(c) if act == "sigmoid" else torch.tanh(c))
    return torch.stack(cols, dim=-1)


def encode_angle(theta: float) -> tuple[float, float]:
    """θ (rad, orientación en [0, π)) -> (sin 2θ, cos 2θ)."""
    return math.sin(2.0 * theta), math.cos(2.0 * theta)


def decode_angle(sin2: float, cos2: float) -> float:
    """(sin 2θ, cos 2θ) -> θ en [0, π) (rad)."""
    a = 0.5 * math.atan2(float(sin2), float(cos2))
    if a < 0:
        a += math.pi
    return a


def angle_error_deg(sin2_a, cos2_a, sin2_b, cos2_b) -> float:
    """Distancia angular (grados, en [0, 90]) entre dos orientaciones dadas por
    sus pares (sin 2θ, cos 2θ)."""
    d = abs(decode_angle(sin2_a, cos2_a) - decode_angle(sin2_b, cos2_b)) % math.pi
    d = min(d, math.pi - d)
    return math.degrees(d)


# Categoría discreta derivada de los descriptores: sirve para reutilizar la grilla
# de "confusión" y los filtros (aciertos/fallos/ambiguos) de la pestaña Probar sin
# rediseñarla. "vacío" cuando la tinta es ~0; si no, recta/curva por `straightness`
# y continua/entrecortada por `continuity`.
CATEGORIES: list[str] = [
    "recta·continua", "recta·entrecortada",
    "curva·continua", "curva·entrecortada",
    "vacío",
]
EMPTY_CATEGORY = 4


def category_index(vec) -> int:
    """vec en rango natural -> índice en CATEGORIES (umbral 0.5; vacío si ink < EMPTY_INK)."""
    if float(vec[IDX["ink"]]) < EMPTY_INK:
        return EMPTY_CATEGORY
    straight = float(vec[IDX["straightness"]]) >= 0.5
    continuous = float(vec[IDX["continuity"]]) >= 0.5
    return (0 if straight else 2) + (0 if continuous else 1)


def boundary_margin(vec) -> float:
    """Distancia a la frontera de decisión de la categoría, en [0, 1].

    Análogo del `margin` de la clasificación: pequeño = muestra ambigua (los
    descriptores rectitud/continuidad están cerca de 0.5).
    """
    s = abs(float(vec[IDX["straightness"]]) - 0.5)
    c = abs(float(vec[IDX["continuity"]]) - 0.5)
    return 2.0 * min(s, c)
