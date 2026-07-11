"""Entrenamiento del dimensionador (CNN sobre ventanas).

Métricas: cross-entropy y accuracy en train/val por época (y por lote cada
`log_every` pasos), test final de MNIST al terminar.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swnist.data.registry import build_dataset
from swnist.experiments.backup import backup_experiment
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.dimensionador import Dimensionador
from swnist.repro import set_seed
from .common import get_device, load_init_weights, make_loaders, save_checkpoint


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    if n == 0:  # split de val vacío (dataset minúsculo)
        return 0.0, 0.0
    return total_loss / n, correct / n


def run_training(exp_id: str, config: dict, registry: ExperimentRegistry, stop_event=None):
    tr = config["training"]
    gen = set_seed(tr["seed"])
    device = get_device()

    dataset = build_dataset(config["dataset"]["name"], config["dataset"]["params"],
                            train=True, seed=tr["seed"])
    train_loader, val_loader = make_loaders(dataset, tr["val_fraction"], tr["batch_size"], gen)

    model = Dimensionador.from_config(config["model"]).to(device)
    if config.get("init_from"):
        load_init_weights(model, registry, config["init_from"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = registry.checkpoints_dir(exp_id)
    registry.mark_running(exp_id)
    best_val_acc, stopped = -1.0, False

    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * y.numel()
            run_correct += (logits.argmax(1) == y).sum().item()
            run_n += y.numel()
            if step % tr["log_every"] == 0:
                registry.log_metrics(exp_id, {
                    "type": "batch", "epoch": epoch, "step": step,
                    "loss": round(loss.item(), 5),
                    "acc": round((logits.argmax(1) == y).float().mean().item(), 5),
                })
        if stopped:
            break

        val_loss, val_acc = evaluate(model, val_loader, device)
        registry.log_metrics(exp_id, {
            "type": "epoch", "epoch": epoch,
            "train_loss": round(run_loss / run_n, 5),
            "train_acc": round(run_correct / run_n, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "seconds": round(time.time() - t0, 2),
        })
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, config)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, config,
                            extra={"val_acc": val_acc})
            registry.update_status(exp_id, best={"epoch": epoch, "val_acc": round(val_acc, 5),
                                                 "val_loss": round(val_loss, 5)})

    # Evaluación final en el test de MNIST (mismo tipo de dataset)
    test_ds = build_dataset(config["dataset"]["name"], config["dataset"]["params"],
                            train=False, seed=tr["seed"])
    test_loader = DataLoader(test_ds, batch_size=tr["batch_size"], num_workers=0)
    test_loss, test_acc = evaluate(model, test_loader, device)

    registry.mark_finished(exp_id, status="stopped" if stopped else "completed",
                           final_test={"loss": round(test_loss, 5), "acc": round(test_acc, 5)})
    backup_experiment(exp_id, experiments_root=registry.root)
