"""Utilidades compartidas de entrenamiento: loaders reproducibles y checkpoints."""

import torch
from torch.utils.data import DataLoader, random_split


def make_loaders(dataset, val_fraction: float, batch_size: int, generator: torch.Generator):
    n_val = int(len(dataset) * val_fraction)
    # Datasets pequeños (p. ej. custom filtrados): garantiza al menos 1 muestra de val
    if val_fraction > 0 and n_val == 0 and len(dataset) > 1:
        n_val = 1
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


def load_init_weights(model, registry, init_from: str, device) -> None:
    """Re-entrenamiento: carga en `model` los pesos best.pt del experimento origen."""
    ckpt_path = registry.checkpoints_dir(init_from) / "best.pt"
    if not ckpt_path.exists():
        raise ValueError(f"Re-entrenamiento: no existe checkpoint best.pt en "
                         f"{init_from!r} (¿experimento incompleto o eliminado?).")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
