"""Utilidades compartidas de entrenamiento: loaders reproducibles y checkpoints."""

import torch
from torch.utils.data import DataLoader, random_split


def make_loaders(dataset, val_fraction: float, batch_size: int, generator: torch.Generator):
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    # num_workers=0: seguro en Windows y reproducible
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def save_checkpoint(path, model, optimizer, epoch: int, config: dict, extra: dict | None = None):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config,
        "epoch": epoch,
        "config": config,
        **(extra or {}),
    }, path)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
