"""Entrenamiento del secuenciador (RNN retroalimentada sobre secuencias de ventanas).

El dimensionador se carga desde el checkpoint `best.pt` del experimento indicado en
`config["dimensionador_experiment"]`; su `window_size` se impone al dataset de
secuencias para que las ventanas coincidan con lo que la CNN sabe procesar.

Métricas: pérdida/accuracy (del último paso) en train/val por época, accuracy por
paso de la secuencia en val (muestra cómo mejora la predicción a medida que la red
"ve" más partes de la imagen), y test final.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swnist.data.registry import build_dataset
from swnist.experiments.backup import backup_experiment
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.dimensionador import Dimensionador
from swnist.models.secuenciador import Secuenciador
from swnist.repro import set_seed
from .common import get_device, make_loaders, save_checkpoint


def load_dimensionador(registry: ExperimentRegistry, exp_id: str, device) -> Dimensionador:
    ckpt_path = registry.checkpoints_dir(exp_id) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Dimensionador.from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def compute_features(dim_model, windows, freeze: bool):
    """windows (B, T, 1, ws, ws) -> features (B, T, D)."""
    B, T = windows.shape[:2]
    flat = windows.reshape(B * T, *windows.shape[2:])
    if freeze:
        with torch.no_grad():
            feats = dim_model.features(flat)
    else:
        feats = dim_model.features(flat)
    return feats.reshape(B, T, -1)


def sequence_loss(logits_seq, labels, mode: str, criterion):
    """logits_seq (B, T, C), labels (B)."""
    T = logits_seq.shape[1]
    if mode == "final":
        return criterion(logits_seq[:, -1], labels)
    losses = torch.stack([criterion(logits_seq[:, t], labels) for t in range(T)])
    if mode == "all":
        return losses.mean()
    if mode == "ramp":  # peso creciente con el paso
        w = torch.arange(1, T + 1, device=losses.device, dtype=losses.dtype)
        return (losses * w).sum() / w.sum()
    raise ValueError(f"loss_mode desconocido: {mode!r}")


@torch.no_grad()
def evaluate(seq_model, dim_model, loader, device, loss_mode, freeze) -> dict:
    seq_model.eval()
    dim_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches, correct_final, n = 0.0, 0, 0, 0
    per_step_correct = None
    for windows, positions, labels in loader:
        windows, positions, labels = windows.to(device), positions.to(device), labels.to(device)
        feats = compute_features(dim_model, windows, freeze=True)
        logits_seq, _ = seq_model(feats, positions)
        total_loss += sequence_loss(logits_seq, labels, loss_mode, criterion).item()
        n_batches += 1
        preds = logits_seq.argmax(-1)  # (B, T)
        correct_final += (preds[:, -1] == labels).sum().item()
        step_correct = (preds == labels.unsqueeze(1)).sum(0)  # (T,)
        per_step_correct = step_correct if per_step_correct is None else per_step_correct + step_correct
        n += labels.numel()
    return {
        "loss": total_loss / n_batches,
        "acc": correct_final / n,
        "per_step_acc": [round(c / n, 5) for c in per_step_correct.tolist()],
    }


def run_training(exp_id: str, config: dict, registry: ExperimentRegistry, stop_event=None):
    tr = config["training"]
    gen = set_seed(tr["seed"])
    device = get_device()

    dim_exp = config.get("dimensionador_experiment")
    if not dim_exp:
        raise ValueError("Falta 'dimensionador_experiment': entrena primero un dimensionador "
                         "y selecciónalo en el formulario.")
    freeze = bool(config.get("freeze_dimensionador", True))
    dim_model = load_dimensionador(registry, dim_exp, device)
    dim_model.eval() if freeze else dim_model.train()

    # El tamaño de ventana lo dicta el dimensionador entrenado
    ds_params = dict(config["dataset"]["params"])
    ds_params["window_size"] = dim_model.config["window_size"]
    dataset = build_dataset(config["dataset"]["name"], ds_params, train=True, seed=tr["seed"])
    train_loader, val_loader = make_loaders(dataset, tr["val_fraction"], tr["batch_size"], gen)

    seq_model = Secuenciador(
        feature_dim=dim_model.config["feature_dim"], **config["model"]
    ).to(device)
    params = list(seq_model.parameters()) + ([] if freeze else list(dim_model.parameters()))
    optimizer = torch.optim.Adam(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    loss_mode = tr.get("loss_mode", "all")

    ckpt_dir = registry.checkpoints_dir(exp_id)
    registry.mark_running(exp_id)
    best_val_acc, stopped = -1.0, False

    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        seq_model.train()
        run_loss, run_correct, run_n, n_batches = 0.0, 0, 0, 0
        for step, (windows, positions, labels) in enumerate(train_loader, 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            windows, positions, labels = windows.to(device), positions.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = compute_features(dim_model, windows, freeze)
            logits_seq, _ = seq_model(feats, positions)
            loss = sequence_loss(logits_seq, labels, loss_mode, criterion)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            n_batches += 1
            final_preds = logits_seq[:, -1].argmax(-1)
            run_correct += (final_preds == labels).sum().item()
            run_n += labels.numel()
            if step % tr["log_every"] == 0:
                registry.log_metrics(exp_id, {
                    "type": "batch", "epoch": epoch, "step": step,
                    "loss": round(loss.item(), 5),
                    "acc": round((final_preds == labels).float().mean().item(), 5),
                })
        if stopped:
            break

        val = evaluate(seq_model, dim_model, val_loader, device, loss_mode, freeze)
        registry.log_metrics(exp_id, {
            "type": "epoch", "epoch": epoch,
            "train_loss": round(run_loss / n_batches, 5),
            "train_acc": round(run_correct / run_n, 5),
            "val_loss": round(val["loss"], 5),
            "val_acc": round(val["acc"], 5),
            "val_per_step_acc": val["per_step_acc"],
            "seconds": round(time.time() - t0, 2),
        })
        save_checkpoint(ckpt_dir / "last.pt", seq_model, optimizer, epoch, config)
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            save_checkpoint(ckpt_dir / "best.pt", seq_model, optimizer, epoch, config,
                            extra={"val_acc": val["acc"],
                                   "dimensionador_experiment": dim_exp})
            registry.update_status(exp_id, best={"epoch": epoch, "val_acc": round(val["acc"], 5),
                                                 "val_loss": round(val["loss"], 5),
                                                 "val_per_step_acc": val["per_step_acc"]})

    test_ds = build_dataset(config["dataset"]["name"], ds_params, train=False, seed=tr["seed"])
    test_loader = DataLoader(test_ds, batch_size=tr["batch_size"], num_workers=0)
    test = evaluate(seq_model, dim_model, test_loader, device, loss_mode, freeze)

    registry.mark_finished(exp_id, status="stopped" if stopped else "completed",
                           final_test={"loss": round(test["loss"], 5),
                                       "acc": round(test["acc"], 5),
                                       "per_step_acc": test["per_step_acc"]})
    backup_experiment(exp_id)
