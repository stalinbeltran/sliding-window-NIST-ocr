"""Piezas comunes del entrenamiento: split train/val, loaders reproducibles,
checkpoints e inicialización desde otro experimento (`init_from`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from swnist.repro import seeded_generator


def split_train_val(
    dataset: Dataset, val_fraction: float, seed: int
) -> tuple[Dataset, Dataset | None]:
    """Separa validación del train con semilla fija."""
    total = len(dataset)  # type: ignore[arg-type]
    val_size = int(round(total * val_fraction))
    if val_size <= 0 or val_size >= total:
        return dataset, None
    train_size = total - val_size
    generator = seeded_generator(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    return train_set, val_set


def make_loader(
    dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    """DataLoader reproducible (num_workers=0 por Windows, generador seedeado)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=seeded_generator(seed) if shuffle else None,
    )


def subset_indices(dataset: Dataset) -> list[int] | None:
    return list(dataset.indices) if isinstance(dataset, Subset) else None


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "best": best,
        },
        path,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_init_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Continúa desde los pesos de otro experimento (el optimizador parte de cero)."""
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
