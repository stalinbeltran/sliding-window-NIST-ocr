"""Entrenamiento del dimensionador (CNN sobre ventanas de trazo).

El dimensionador ya no clasifica dígitos: regresa un vector de descriptores
geométricos (ver ``swnist.descriptors``) sobre el dataset sintético de rectas/curvas.

Pérdida: MSE por canal sobre las salidas activadas. Métrica principal (registrada
en la clave `val_acc` para que el registro/UI y la selección del mejor checkpoint
sigan igual): un **score combinado ∈ [0, 1]** = media de (1 − error) por descriptor,
con el ángulo medido como distancia angular. En `metrics.jsonl` se añaden detalles
por época (error de ángulo en grados, MAE de rectitud/continuidad).
"""

import math
import time

import torch
from torch.utils.data import DataLoader

from swnist.data.registry import build_dataset
from swnist.descriptors import IDX, activate
from swnist.experiments.backup import backup_experiment
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.dimensionador import Dimensionador
from swnist.repro import set_seed
from .common import get_device, load_init_weights, make_loaders, save_checkpoint

_SI, _CI = IDX["angle_sin2"], IDX["angle_cos2"]


def _angle_err_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distancia angular (grados, [0, 90]) por muestra entre pred y target activados."""
    pa = 0.5 * torch.atan2(pred[:, _SI], pred[:, _CI])
    ta = 0.5 * torch.atan2(target[:, _SI], target[:, _CI])
    d = torch.remainder(pa - ta, math.pi)
    d = torch.minimum(d, math.pi - d)
    return torch.rad2deg(d)


_SCALAR = [IDX[n] for n in ("straightness", "horizontal", "vertical", "continuity", "ink")]


def masked_mse(pred_act: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE por canal supervisado (mask=1); las ventanas vacías no penalizan geometría."""
    return ((pred_act - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def _batch_metrics(pred_act: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Sumas/contadores sobre el lote (respetando la máscara) para promediar por época."""
    e = (pred_act - target).abs()
    ang = _angle_err_deg(pred_act, target)  # [0, 90]
    e_sc, m_sc = e[:, _SCALAR], mask[:, _SCALAR]
    m_ang = mask[:, _SI]
    # Score por muestra: media de (1 - error) SOLO sobre los canales supervisados
    # (una ventana vacía puntúa solo por su tinta; el ángulo cuenta como un término).
    good = ((1.0 - e_sc) * m_sc).sum(1) + (1.0 - ang / 90.0) * m_ang
    cnt = m_sc.sum(1) + m_ang
    score = (good / cnt.clamp_min(1.0))
    m_str, m_ink = mask[:, IDX["straightness"]], mask[:, IDX["ink"]]
    return {
        "score": score.sum().item(), "n": target.shape[0],
        "angle_sum": (ang * m_ang).sum().item(), "angle_n": m_ang.sum().item(),
        "straightness_sum": (e[:, IDX["straightness"]] * m_str).sum().item(),
        "continuity_sum": (e[:, IDX["continuity"]] * m_str).sum().item(),
        "geom_n": m_str.sum().item(),
        "ink_sum": (e[:, IDX["ink"]] * m_ink).sum().item(), "ink_n": m_ink.sum().item(),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, dict]:
    model.eval()
    total_loss, total_terms = 0.0, 0.0
    keys = ("score", "n", "angle_sum", "angle_n", "straightness_sum",
            "continuity_sum", "geom_n", "ink_sum", "ink_n")
    agg = {k: 0.0 for k in keys}
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred_act = activate(model(x))
        total_loss += ((pred_act - y) ** 2 * m).sum().item()
        total_terms += m.sum().item()
        bm = _batch_metrics(pred_act, y, m)
        for k in keys:
            agg[k] += bm[k]
    n = agg["n"]
    if n == 0:  # split de val vacío (dataset minúsculo)
        return 0.0, 0.0, {}
    loss = total_loss / max(total_terms, 1.0)
    details = {
        "angle_err_deg": round(agg["angle_sum"] / max(agg["angle_n"], 1), 4),
        "straightness_mae": round(agg["straightness_sum"] / max(agg["geom_n"], 1), 4),
        "continuity_mae": round(agg["continuity_sum"] / max(agg["geom_n"], 1), 4),
        "ink_mae": round(agg["ink_sum"] / max(agg["ink_n"], 1), 4),
    }
    return loss, agg["score"] / n, details


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

    ckpt_dir = registry.checkpoints_dir(exp_id)
    registry.mark_running(exp_id)
    best_val_score, stopped = -1.0, False

    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        model.train()
        run_loss, run_score, run_n = 0.0, 0.0, 0
        for step, (x, y, m) in enumerate(train_loader, 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            pred_act = activate(model(x))
            loss = masked_mse(pred_act, y, m)
            loss.backward()
            optimizer.step()

            bm = _batch_metrics(pred_act.detach(), y, m)
            run_loss += loss.item() * y.shape[0]
            run_score += bm["score"]
            run_n += y.shape[0]
            if step % tr["log_every"] == 0:
                registry.log_metrics(exp_id, {
                    "type": "batch", "epoch": epoch, "step": step,
                    "loss": round(loss.item(), 5),
                    "acc": round(bm["score"] / bm["n"], 5),
                })
        if stopped:
            break

        val_loss, val_score, details = evaluate(model, val_loader, device)
        registry.log_metrics(exp_id, {
            "type": "epoch", "epoch": epoch,
            "train_loss": round(run_loss / run_n, 5),
            "train_acc": round(run_score / run_n, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_score, 5),
            "seconds": round(time.time() - t0, 2),
            **details,
        })
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, config)
        if val_score > best_val_score:
            best_val_score = val_score
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, config,
                            extra={"val_acc": val_score})
            registry.update_status(exp_id, best={"epoch": epoch, "val_acc": round(val_score, 5),
                                                 "val_loss": round(val_loss, 5)})

    # Evaluación final en el split de test (mismo tipo de dataset)
    test_ds = build_dataset(config["dataset"]["name"], config["dataset"]["params"],
                            train=False, seed=tr["seed"])
    test_loader = DataLoader(test_ds, batch_size=tr["batch_size"], num_workers=0)
    test_loss, test_score, _ = evaluate(model, test_loader, device)

    registry.mark_finished(exp_id, status="stopped" if stopped else "completed",
                           final_test={"loss": round(test_loss, 5), "acc": round(test_score, 5)})
    backup_experiment(exp_id, experiments_root=registry.root)
