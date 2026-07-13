"""Entrenamiento de la CNN de features sobre MNIST."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from swnist.data.registry import build_dataset
from swnist.experiments.backup import backup_experiment
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.features_cnn import FeaturesCNN
from swnist.repro import pick_device, set_seed
from swnist.training.common import (
    load_init_weights,
    make_loader,
    save_checkpoint,
    split_train_val,
)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
    if seen == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / seen, "acc": correct / seen}


def run_training(
    config: dict[str, Any],
    exp_id: str,
    registry: ExperimentRegistry,
    *,
    stop_event: threading.Event | None = None,
    backups_dir: Path | str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Entrena, registra métricas por lote/época, guarda checkpoints y respalda.

    Cualquier fallo (también al construir dataset/modelo) deja el experimento en
    estado 'failed' con la razón, y se re-lanza.
    """
    device = device or pick_device()
    training = config["training"]
    best: dict[str, Any] | None = None
    stopped = False
    step = 0
    try:
        set_seed(config["seed"])

        dataset = build_dataset(
            config["dataset"]["name"], train=True, **config["dataset"]["params"]
        )
        train_set, val_set = split_train_val(dataset, training["val_fraction"], config["seed"])
        train_loader = make_loader(
            train_set, batch_size=training["batch_size"], shuffle=True, seed=config["seed"]
        )
        val_loader = (
            make_loader(
                val_set, batch_size=training["batch_size"], shuffle=False, seed=config["seed"]
            )
            if val_set is not None
            else None
        )

        model = FeaturesCNN.from_config(config["model"]).to(device)
        if config.get("init_from"):
            load_init_weights(model, registry.checkpoint_path(config["init_from"]))

        optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
        criterion = nn.CrossEntropyLoss()

        registry.update_status(
            exp_id,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            device=str(device),
            num_params=sum(p.numel() for p in model.parameters()),
            layer_shapes=model.layer_shapes(),
        )

        for epoch in range(1, training["epochs"] + 1):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            running_correct = 0
            running_seen = 0

            for images, labels in train_loader:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                step += 1
                batch_correct = int((logits.argmax(dim=1) == labels).sum().item())
                running_loss += float(loss.item()) * labels.size(0)
                running_correct += batch_correct
                running_seen += labels.size(0)

                if step % training["log_every"] == 0:
                    registry.log_metric(
                        exp_id,
                        {
                            "type": "batch",
                            "epoch": epoch,
                            "step": step,
                            "loss": float(loss.item()),
                            "acc": batch_correct / labels.size(0),
                        },
                    )

            if stopped:
                break

            train_loss = running_loss / max(running_seen, 1)
            train_acc = running_correct / max(running_seen, 1)
            val_metrics = (
                evaluate(model, val_loader, criterion, device)
                if val_loader is not None
                else {"loss": train_loss, "acc": train_acc}
            )
            registry.log_metric(
                exp_id,
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "seconds": time.time() - epoch_start,
                },
            )

            if best is None or val_metrics["acc"] > best["val_acc"]:
                best = {
                    "epoch": epoch,
                    "val_acc": val_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                }
                save_checkpoint(
                    registry.checkpoint_path(exp_id, "best.pt"),
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best=best,
                )
            save_checkpoint(
                registry.checkpoint_path(exp_id, "last.pt"),
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best=best,
            )
            registry.update_status(exp_id, best=best, epochs_done=epoch)

        final_test = None
        if not stopped:
            test_set = build_dataset(
                config["dataset"]["name"], train=False, **config["dataset"]["params"]
            )
            test_loader = make_loader(
                test_set,
                batch_size=training["batch_size"],
                shuffle=False,
                seed=config["seed"],
            )
            final_test = evaluate(model, test_loader, criterion, device)

        status = registry.update_status(
            exp_id,
            status="stopped" if stopped else "completed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            best=best,
            final_test=final_test,
        )
    except Exception as exc:  # noqa: BLE001 - se registra y se re-lanza
        registry.update_status(
            exp_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    backup_experiment(registry.path(exp_id), backups_dir=backups_dir)
    return status
