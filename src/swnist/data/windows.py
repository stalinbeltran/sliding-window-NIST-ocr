"""Utilidades de ventana deslizante sobre imágenes 2D.

El recorrido de la ventana lo define la trayectoria por el trazo del carácter
(ver `swnist.data.trajectory`): el barrido raster de la imagen se eliminó
(2026-07-12) porque no aportaba señal —los desplazamientos eran constantes y no
describían la forma.
"""

import torch


def extract_window(image: torch.Tensor, top: int, left: int, window_size: int) -> torch.Tensor:
    """image (1, H, W) -> ventana (1, ws, ws)."""
    return image[:, top : top + window_size, left : left + window_size]


def normalize_position(top: int, left: int, image_size: int, window_size: int) -> tuple[float, float]:
    """(top, left) -> (x, y) en [0, 1] relativo al rango válido de la esquina."""
    denom = max(image_size - window_size, 1)
    return left / denom, top / denom
