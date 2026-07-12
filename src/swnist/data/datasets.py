"""Datasets sobre MNIST para el secuenciador (secuencias de ventanas).

El dataset del dimensionador (rectas/curvas sintéticas) vive en
``swnist.data.synthstrokes``. Aquí quedan solo los datasets de secuencias que
recorren dígitos MNIST reales. Todos son deterministas dado (params, seed).
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from swnist import DATA_DIR
from .strokes import StrokeUniformizer
from .trajectory import contour_positions
from .windows import extract_window, grid_positions, normalize_position

IMAGE_SIZE = 28

_MEAN, _STD = 0.1307, 0.3081

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_MEAN,), (_STD,)),
])


def load_mnist(train: bool) -> datasets.MNIST:
    return datasets.MNIST(str(DATA_DIR), train=train, download=True, transform=_TRANSFORM)


class MnistSlidingSequences(Dataset):
    """Secuencia de ventanas (recorrido raster) + posiciones normalizadas.

    Compatible con: secuenciador. Devuelve:
      windows   (T, 1, ws, ws)
      positions (T, 2)  — (x, y) en [0, 1]
      label     int
    """

    def __init__(
        self,
        train: bool = True,
        window_size: int = 14,
        stride: int = 7,
        seed: int = 0,
        stroke_width: int = 0,
    ):
        self.base = load_mnist(train)
        self.window_size = window_size
        self.stride = int(stride)
        self.stroke_width = int(stroke_width)
        self._strokes = StrokeUniformizer(stroke_width)
        self.positions = grid_positions(IMAGE_SIZE, window_size, stride)

    @property
    def sequence_length(self) -> int:
        return len(self.positions)

    def __len__(self):
        return len(self.base)

    def trajectory(self, idx) -> list[tuple[int, int]]:
        """Posiciones (top, left) del recorrido (fijo: no depende de la muestra)."""
        return self.positions

    def __getitem__(self, idx):
        image, label = self.base[idx]
        image = self._strokes(image, idx)
        windows, coords = [], []
        for top, left in self.trajectory(idx):
            windows.append(extract_window(image, top, left, self.window_size))
            coords.append(normalize_position(top, left, IMAGE_SIZE, self.window_size))
        return (
            torch.stack(windows),
            torch.tensor(coords, dtype=torch.float32),
            label,
        )

    def display_item(self, idx):
        """La muestra visible es la imagen completa que recorre la ventana."""
        image, label = self.base[idx]
        return self._strokes(image, idx), label


class MnistContourSequences(Dataset):
    """Secuencia de ventanas que sigue la 'forma' (trazo) del carácter.

    Compatible con: secuenciador. Misma estructura de muestra que
    MnistSlidingSequences, pero la trayectoria se deriva de cada imagen
    (esqueleto del trazo -> camino ordenado -> num_steps posiciones, ver
    swnist.data.trajectory) en vez de escanear el área completa en raster.
    T = num_steps es fijo para todas las muestras (batching y accuracy por
    paso). Determinista dado (params, imagen): no usa RNG.
    """

    def __init__(
        self,
        train: bool = True,
        window_size: int = 14,
        num_steps: int = 12,
        seed: int = 0,
        stroke_width: int = 0,
    ):
        self.base = load_mnist(train)
        self.window_size = int(window_size)
        self.num_steps = int(num_steps)
        self.stroke_width = int(stroke_width)
        self._strokes = StrokeUniformizer(stroke_width)
        self._traj_cache: dict[int, list[tuple[int, int]]] = {}

    def _image(self, idx) -> tuple[torch.Tensor, int]:
        """Imagen con el trazo uniformizado (si stroke_width > 0): la
        trayectoria y las ventanas deben salir de la MISMA imagen."""
        image, label = self.base[idx]
        return self._strokes(image, idx), label

    @property
    def sequence_length(self) -> int:
        return self.num_steps

    def __len__(self):
        return len(self.base)

    def trajectory(self, idx) -> list[tuple[int, int]]:
        """Posiciones (top, left) de la ventana para la muestra idx (cacheadas:
        el esqueleto solo se calcula una vez por imagen)."""
        if idx not in self._traj_cache:
            image, _ = self._image(idx)
            self._traj_cache[idx] = contour_positions(
                image, self.window_size, self.num_steps, IMAGE_SIZE)
        return self._traj_cache[idx]

    def __getitem__(self, idx):
        image, label = self._image(idx)
        windows, coords = [], []
        for top, left in self.trajectory(idx):
            windows.append(extract_window(image, top, left, self.window_size))
            coords.append(normalize_position(top, left, IMAGE_SIZE, self.window_size))
        return (
            torch.stack(windows),
            torch.tensor(coords, dtype=torch.float32),
            label,
        )

    def display_item(self, idx):
        """La muestra visible es la imagen completa que recorre la ventana."""
        return self._image(idx)
