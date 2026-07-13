"""Dataset MNIST: imágenes completas 28x28 + etiqueta 0-9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


class MnistDigits(Dataset):
    """Imágenes MNIST completas, normalizadas a [0, 1] con shape (1, 28, 28).

    Params:
      - `limit`: usa solo las primeras N muestras del split (None = todas).
    """

    IMAGE_SIZE = 28
    NUM_CLASSES = 10

    def __init__(self, *, train: bool = True, limit: int | None = None, root: Path | None = None):
        self.train = train
        self.limit = limit
        self.base = datasets.MNIST(
            root=str(root or DATA_ROOT),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        total = len(self.base)
        self.length = total if limit is None else min(int(limit), total)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[idx]
        return image, int(label)

    def display_item(self, idx: int) -> np.ndarray:
        """Imagen 28x28 en [0, 1] para las miniaturas de la web app."""
        image, _ = self.base[idx]
        return image.squeeze(0).numpy()

    @staticmethod
    def defaults() -> dict[str, Any]:
        return {"limit": None}
