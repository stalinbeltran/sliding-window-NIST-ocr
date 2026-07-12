"""Uniformización del grosor del trazo de un carácter tipo NIST/MNIST.

Un mismo dígito puede venir escrito con trazo grueso o delgado; esta
transformación lo redibuja con un grosor constante para que ambos terminen
viéndose iguales: la tinta se adelgaza a su esqueleto de ~1 px (Zhang-Suen,
reutilizado de swnist.data.trajectory) y se vuelve a "entintar" alrededor del
esqueleto hasta el grosor pedido (stroke_width, en píxeles), con un borde
suave de ~1 px que imita el anti-alias de MNIST. Es determinista dado
(imagen, stroke_width): no usa RNG.
"""

import numpy as np
import torch

from .trajectory import ink_mask, skeletonize

# Estadísticas de normalización de MNIST (la imagen llega/sale normalizada).
_MEAN, _STD = 0.1307, 0.3081


def _uniform_stroke_gray(image: torch.Tensor, stroke_width: int) -> np.ndarray | None:
    """Gris en [0, 1] (H, W) del carácter redibujado, o None si no hay tinta."""
    skel = skeletonize(ink_mask(image))
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return None
    h, w = skel.shape
    rows, cols = np.mgrid[0:h, 0:w]
    # Distancia euclidiana de cada píxel al punto más cercano del esqueleto
    # (fuerza bruta: 28×28 × |esqueleto| es trivial).
    d2 = (rows[..., None] - pts[:, 0]) ** 2 + (cols[..., None] - pts[:, 1]) ** 2
    dist = np.sqrt(d2.min(axis=-1))
    # Intensidad 1 dentro del medio-grosor y caída lineal de 1 px en el borde.
    return np.clip(stroke_width / 2 + 0.5 - dist, 0.0, 1.0)


def uniform_stroke(image: torch.Tensor, stroke_width: int) -> torch.Tensor:
    """Redibuja el carácter con trazo de ~stroke_width px de grosor uniforme.

    image (1, H, W) normalizada -> imagen normalizada del mismo tamaño. Un
    trazo grueso y uno delgado del mismo esqueleto convergen a la misma forma.
    Con stroke_width <= 0 o sin tinta detectable la imagen vuelve intacta
    (una ventana vacía sigue vacía).
    """
    if stroke_width <= 0:
        return image
    gray = _uniform_stroke_gray(image, int(stroke_width))
    if gray is None:
        return image
    return torch.from_numpy(((gray - _MEAN) / _STD).astype(np.float32)).unsqueeze(0)


class StrokeUniformizer:
    """uniform_stroke con caché por índice de muestra (el esqueleto es lo caro).

    La caché guarda el gris cuantizado a uint8 (mismo redondeo que el PNG de
    las miniaturas: indistinguible y 4× más compacto que float32). Con
    stroke_width=0 es un passthrough sin coste ni caché.
    """

    def __init__(self, stroke_width: int = 0):
        self.stroke_width = int(stroke_width)
        self._cache: dict[int, torch.Tensor | None] = {}

    def __call__(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        if self.stroke_width <= 0:
            return image
        if idx not in self._cache:
            gray = _uniform_stroke_gray(image, self.stroke_width)
            self._cache[idx] = None if gray is None else \
                torch.from_numpy(np.round(gray * 255).astype(np.uint8))
        cached = self._cache[idx]
        if cached is None:
            return image
        return ((cached.float() / 255.0 - _MEAN) / _STD).unsqueeze(0)
