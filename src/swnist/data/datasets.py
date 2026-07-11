"""Datasets sobre MNIST para el dimensionador y el secuenciador.

Todos son deterministas dado (params, seed): las ventanas aleatorias se derivan
de una RNG sembrada por (seed, index) para que un experimento sea reproducible.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from swnist import DATA_DIR
from .windows import extract_window, grid_positions, normalize_position

IMAGE_SIZE = 28

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_mnist(train: bool) -> datasets.MNIST:
    return datasets.MNIST(str(DATA_DIR), train=train, download=True, transform=_TRANSFORM)


class MnistFull(Dataset):
    """Imágenes MNIST completas, adaptables a cualquier tamaño de ventana.

    Compatible con: dimensionador. Con los defaults (window_size=28,
    windows_per_image=1) cada muestra es la imagen completa 28×28. Con
    window_size < 28 cada muestra es una ventana aleatoria de la imagen
    (determinista dado (seed, idx)), con `windows_per_image` muestras por
    imagen; la muestra visible (display_item) sigue siendo la imagen completa.
    """

    def __init__(
        self,
        train: bool = True,
        seed: int = 0,
        window_size: int = IMAGE_SIZE,
        windows_per_image: int = 1,
    ):
        self.base = load_mnist(train)
        self.window_size = int(window_size)
        self.windows_per_image = int(windows_per_image)
        self.seed = seed

    def __len__(self):
        return len(self.base) * self.windows_per_image

    def __getitem__(self, idx):
        image, label = self.base[idx // self.windows_per_image]
        if self.window_size >= IMAGE_SIZE:
            return image, label  # (image (1,28,28), label)
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        last = IMAGE_SIZE - self.window_size
        top = int(torch.randint(0, last + 1, (1,), generator=gen).item())
        left = int(torch.randint(0, last + 1, (1,), generator=gen).item())
        return extract_window(image, top, left, self.window_size), label

    def display_item(self, idx):
        """(imagen mostrable, etiqueta): la imagen completa de la que sale la ventana."""
        return self.base[idx // self.windows_per_image]


class MnistWindows(MnistFull):
    """Ventanas aleatorias de imágenes MNIST, etiquetadas con el dígito de origen.

    Compatible con: dimensionador. Misma generación de ventanas que MnistFull,
    pero la muestra visible es la ventana misma (lo que recibe la NN).
    """

    def __init__(
        self,
        train: bool = True,
        window_size: int = 14,
        windows_per_image: int = 4,
        seed: int = 0,
    ):
        super().__init__(train=train, seed=seed, window_size=window_size,
                         windows_per_image=windows_per_image)

    def display_item(self, idx):
        """La muestra visible es la ventana misma (lo que recibe la NN)."""
        return self[idx]


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
    ):
        self.base = load_mnist(train)
        self.window_size = window_size
        self.positions = grid_positions(IMAGE_SIZE, window_size, stride)

    @property
    def sequence_length(self) -> int:
        return len(self.positions)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        windows, coords = [], []
        for top, left in self.positions:
            windows.append(extract_window(image, top, left, self.window_size))
            coords.append(normalize_position(top, left, IMAGE_SIZE, self.window_size))
        return (
            torch.stack(windows),
            torch.tensor(coords, dtype=torch.float32),
            label,
        )

    def display_item(self, idx):
        """La muestra visible es la imagen completa que recorre la ventana."""
        return self.base[idx]
