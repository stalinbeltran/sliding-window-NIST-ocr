"""Runner de evaluaciones: aplica una NN entrenada a un dataset y registra el
resultado por muestra (para filtrar aciertos/fallos/ambigüedades en la web app).

Config: {"experiment", "dataset": {"name", "params"}, "split": "train"|"test",
         "limit": int|None, "batch_size": int, "seed": int}

El índice de cada resultado es el índice dentro del dataset construido con esos
mismos (params, split, seed), de modo que las miniaturas, el predict y los
datasets custom guardados desde un filtro refieren siempre a la misma muestra.
"""

from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from swnist.data.registry import build_dataset
from swnist.descriptors import (CATEGORIES, IDX, NAMES, activate,
                                angle_error_deg, boundary_margin, category_index)
from swnist.experiments.registry import ExperimentRegistry
from swnist.models.dimensionador import Dimensionador
from swnist.models.secuenciador import Secuenciador
from swnist.training.common import get_device
from swnist.training.train_secuenciador import compute_features
from swnist.validation import sequence_effective_params

_SI, _CI = IDX["angle_sin2"], IDX["angle_cos2"]

from .registry import EvaluationRegistry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(exp_registry: ExperimentRegistry, exp_id: str, device):
    """Carga best.pt de un experimento → (nn_name, modelo, config del experimento)."""
    exp = exp_registry.get_experiment(exp_id)
    nn_name = exp["config"]["nn"]
    ckpt_path = exp_registry.checkpoints_dir(exp_id) / "best.pt"
    if not ckpt_path.exists():
        raise ValueError(f"El experimento {exp_id!r} no tiene checkpoint best.pt.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cls = {"dimensionador": Dimensionador, "secuenciador": Secuenciador}[nn_name]
    model = cls.from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return nn_name, model, exp["config"]


def _stats(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """probs (B, C) -> (pred, conf, margin)."""
    top2 = probs.topk(2, dim=-1)
    pred = top2.indices[:, 0]
    conf = top2.values[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    return pred, conf, margin


def run_evaluation(eval_id: str, config: dict, eval_registry: EvaluationRegistry,
                   exp_registry: ExperimentRegistry, stop_event=None):
    device = get_device()
    seed = int(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 256))

    nn_name, model, exp_cfg = load_model(exp_registry, config["experiment"], device)

    ds_name = config["dataset"]["name"]
    ds_params = dict(config["dataset"].get("params") or {})
    dim_model = None
    if nn_name == "secuenciador":
        _, dim_model, _ = load_model(exp_registry, exp_cfg["dimensionador_experiment"], device)
        # Ventana Y stride los dicta el entrenamiento (la trayectoria es entrada
        # de la red). validate_eval_config ya los fijó en la config; esto cubre
        # también a quien llame al runner directamente.
        ds_params.update(sequence_effective_params(exp_cfg, exp_registry))

    dataset = build_dataset(ds_name, ds_params, train=(config["split"] == "train"), seed=seed)
    limit = config.get("limit")
    if limit:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    total = len(dataset)

    eval_registry.update_status(eval_id, status="running", started_at=_now(),
                                progress={"done": 0, "total": total})
    done, correct_total, stopped = 0, 0, False
    # El dimensionador regresa descriptores: se resume por CATEGORÍA discreta
    # (recta/curva × continua/entrecortada) para reutilizar la grilla de confusión
    # y los filtros. El secuenciador sigue clasificando dígitos.
    is_dim = nn_name == "dimensionador"
    num_classes = len(CATEGORIES) if is_dim else int(model.config.get("num_classes", 10))
    confusion = [[0] * num_classes for _ in range(num_classes)]
    per_step_correct = None
    desc_err_sum = [0.0] * len(NAMES)   # MAE por descriptor (solo dimensionador)
    angle_err_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            records = []
            if is_dim:
                x, y = batch  # y: targets (B, 6)
                pred_desc = activate(model(x.to(device))).cpu()  # (B, 6)
                for i in range(pred_desc.shape[0]):
                    p, t = pred_desc[i], y[i]
                    pcat, tcat = category_index(p), category_index(t)
                    ok = pcat == tcat
                    correct_total += ok
                    confusion[tcat][pcat] += 1
                    ang = angle_error_deg(p[_SI], p[_CI], t[_SI], t[_CI])
                    angle_err_sum += ang
                    errs = (p - t).abs()
                    for k in range(len(NAMES)):
                        desc_err_sum[k] += float(errs[k])
                    records.append({
                        "index": done + i, "label": tcat, "pred": pcat, "correct": ok,
                        "conf": round(1.0 - boundary_margin(p) / 2.0, 4),
                        "margin": round(boundary_margin(p), 4),
                        "angle_err_deg": round(ang, 2),
                        "descriptors": {n: round(float(p[k]), 3) for k, n in enumerate(NAMES)},
                        "target": {n: round(float(t[k]), 3) for k, n in enumerate(NAMES)},
                    })
                done += pred_desc.shape[0]
            else:
                windows, positions, y = batch
                feats = compute_features(dim_model, windows.to(device), freeze=True)
                logits_seq, _ = model(feats, positions.to(device))
                probs_seq = F.softmax(logits_seq, dim=-1)  # (B, T, C)
                probs = probs_seq[:, -1]
                pred, conf, margin = _stats(probs)
                preds_steps = probs_seq.argmax(-1)  # (B, T)
                step_ok = (preds_steps == y.to(device).unsqueeze(1)).sum(0)
                per_step_correct = step_ok if per_step_correct is None \
                    else per_step_correct + step_ok
                extra = [{"pred_steps": ps} for ps in preds_steps.cpu().tolist()]
                y = y.tolist()
                pred, conf, margin = pred.cpu().tolist(), conf.cpu().tolist(), margin.cpu().tolist()
                for i, label in enumerate(y):
                    ok = pred[i] == label
                    correct_total += ok
                    confusion[label][pred[i]] += 1
                    records.append({
                        "index": done + i, "label": label, "pred": pred[i],
                        "correct": ok, "conf": round(conf[i], 4),
                        "margin": round(margin[i], 4), **extra[i],
                    })
                done += len(y)
            eval_registry.append_results(eval_id, records)
            eval_registry.update_status(eval_id, progress={"done": done, "total": total})

    summary = {
        "n": done,
        "acc": round(correct_total / done, 5) if done else None,
        "errors": done - correct_total,
        "confusion": confusion,
    }
    if is_dim and done:
        summary["categories"] = CATEGORIES
        summary["angle_mae_deg"] = round(angle_err_sum / done, 3)
        summary["descriptor_mae"] = {n: round(desc_err_sum[k] / done, 4)
                                     for k, n in enumerate(NAMES)}
    if per_step_correct is not None and done:
        summary["per_step_acc"] = [round(c / done, 5) for c in per_step_correct.tolist()]
    eval_registry.update_status(
        eval_id, status="stopped" if stopped else "completed",
        finished_at=_now(), summary=summary)
