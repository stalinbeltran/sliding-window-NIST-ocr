"""Trayectoria guiada por el contenido: la ventana sigue la 'forma' del carácter.

En vez de escanear el área completa en orden raster, el recorrido se deriva de
cada imagen: se binariza la tinta, se adelgaza el trazo a un esqueleto de ~1 px
(algoritmo Zhang-Suen), se ordenan los píxeles del esqueleto en un camino
(vecino más cercano desde un extremo del trazo) y se remuestrea ese camino a un
número fijo de pasos. La longitud fija T = num_steps mantiene el contrato de
los datasets de secuencias (batching y accuracy por paso necesitan el mismo T
en todas las muestras). Todo es determinista dado (imagen, params): no usa RNG.
"""

import numpy as np
import torch

# Estadísticas de normalización de MNIST: la imagen llega como (x - mean) / std
# y el umbral de tinta se aplica sobre el gris original en [0, 1].
_MEAN, _STD = 0.1307, 0.3081
INK_THRESHOLD = 0.5


def ink_mask(image: torch.Tensor, threshold: float = INK_THRESHOLD) -> np.ndarray:
    """image (1, H, W) normalizada -> máscara bool (H, W) de píxeles con tinta."""
    gray = image.squeeze(0).cpu().numpy() * _STD + _MEAN
    return gray > threshold


def _ring(padded: np.ndarray) -> list[np.ndarray]:
    """Los 8 vecinos P2..P9 (sentido horario desde el norte) de cada píxel."""
    return [
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
        padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
    ]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Adelgazamiento Zhang-Suen: máscara bool -> esqueleto de ~1 px de grosor."""
    img = mask.astype(bool).copy()
    while True:
        changed = False
        for step in (0, 1):
            ring = _ring(np.pad(img, 1))
            p2, p4, p6, p8 = ring[0], ring[2], ring[4], ring[6]
            b = sum(x.astype(np.int8) for x in ring)
            cyc = ring + ring[:1]
            a = sum((~cyc[k] & cyc[k + 1]).astype(np.int8) for k in range(8))
            if step == 0:
                cond = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                cond = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            delete = img & (b >= 2) & (b <= 6) & (a == 1) & cond
            if delete.any():
                img &= ~delete
                changed = True
        if not changed:
            return img


def order_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Ordena los píxeles del esqueleto en un camino que recorre el trazo.

    Empieza en un extremo (píxel con un solo vecino-8; si el trazo es cerrado,
    como un '0', en el píxel superior-izquierdo) y salta siempre al píxel no
    visitado más cercano; si el trazo tiene varias componentes, el salto las
    encadena en vez de dejarlas sin ver.
    """
    pts = np.argwhere(skeleton)
    if len(pts) == 0:
        return []
    degree = sum(s.astype(np.int8) for s in _ring(np.pad(skeleton, 1)))
    endpoints = np.argwhere(skeleton & (degree == 1))
    start = endpoints[0] if len(endpoints) else pts[0]  # argwhere: orden raster, determinista

    path = [(int(start[0]), int(start[1]))]
    rest = pts[~np.all(pts == start, axis=1)].astype(np.int64)
    cur = start.astype(np.int64)
    while len(rest):
        i = int(((rest - cur) ** 2).sum(axis=1).argmin())
        cur = rest[i]
        path.append((int(cur[0]), int(cur[1])))
        rest = np.delete(rest, i, axis=0)
    return path


def resample_path(path: list[tuple[int, int]], num_steps: int) -> list[tuple[int, int]]:
    """Remuestrea el camino a num_steps puntos equiespaciados por longitud de arco."""
    if not path:
        return []
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) == 1:
        return [(int(pts[0, 0]), int(pts[0, 1]))] * num_steps
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, arc[-1], num_steps)
    rows = np.interp(targets, arc, pts[:, 0])
    cols = np.interp(targets, arc, pts[:, 1])
    return [(int(round(r)), int(round(c))) for r, c in zip(rows, cols)]


def contour_positions(
    image: torch.Tensor, window_size: int, num_steps: int, image_size: int | None = None
) -> list[tuple[int, int]]:
    """Posiciones (top, left) de una ventana que sigue el trazo del carácter.

    La ventana se centra en cada punto del camino remuestreado, con la esquina
    recortada al rango válido de la imagen.
    """
    size = int(image.shape[-1]) if image_size is None else image_size
    path = order_path(skeletonize(ink_mask(image)))
    last = size - window_size
    if not path:
        # Sin tinta detectable (caso degenerado): la ventana se queda centrada.
        return [(last // 2, last // 2)] * num_steps
    half = window_size // 2
    return [
        (min(max(r - half, 0), last), min(max(c - half, 0), last))
        for r, c in resample_path(path, num_steps)
    ]
