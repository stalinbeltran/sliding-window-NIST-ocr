"""Utilidades de ventana deslizante sobre imágenes 2D."""

import torch


def grid_positions(image_size: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Posiciones (top, left) en orden raster que cubren la imagen.

    Incluye siempre la última posición pegada al borde aunque no caiga en la grilla,
    para no dejar franjas sin ver.
    """
    last = image_size - window_size
    coords = list(range(0, last + 1, stride))
    if coords[-1] != last:
        coords.append(last)
    return [(y, x) for y in coords for x in coords]


def extract_window(image: torch.Tensor, top: int, left: int, window_size: int) -> torch.Tensor:
    """image (1, H, W) -> ventana (1, ws, ws)."""
    return image[:, top : top + window_size, left : left + window_size]


def normalize_position(top: int, left: int, image_size: int, window_size: int) -> tuple[float, float]:
    """(top, left) -> (x, y) en [0, 1] relativo al rango válido de la esquina."""
    denom = max(image_size - window_size, 1)
    return left / denom, top / denom
