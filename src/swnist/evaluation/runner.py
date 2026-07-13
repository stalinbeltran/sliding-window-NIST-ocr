"""Aplica una NN entrenada a un dataset y registra el resultado de cada muestra."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from swnist.data.builtin import BUILTIN
from swnist.data.registry import build_dataset, store
from swnist.evaluation.registry import EvaluationRegistry
from swnist.experiments.registry import ExperimentRegistry
from swnist.repro import pick_device
from swnist.webapp import inference

BATCH_SIZE = 256


def resolve_base(dataset_name: str, split: str) -> dict[str, Any]:
    """A qué dataset base (y split) apunta cada muestra evaluada.

    Un dataset custom es un subconjunto de índices de un base: para poder guardar un
    filtro como dataset nuevo hay que registrar el índice en el BASE, no en el subconjunto.
    """
    if dataset_name in BUILTIN:
        return {"base": dataset_name, "base_split": split, "map": None}
    definition = store().get(dataset_name)
    if split == "train":  # el subconjunto guardado
        return {
            "base": definition["base"],
            "base_split": definition["split"],
            "map": list(definition["indices"]),
        }
    return {"base": definition["base"], "base_split": "test", "map": None}


def run_evaluation(
    config: dict[str, Any],
    eval_id: str,
    registry: EvaluationRegistry,
    experiments: ExperimentRegistry,
    *,
    stop_event: threading.Event | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or pick_device()
    try:
        model, exp_config = inference.get_model(config["experiment"], experiments)
        model = model.to(device)
        num_classes = exp_config["model"]["num_classes"]

        dataset = build_dataset(config["dataset"], train=(config["split"] == "train"))
        total = len(dataset)
        if config["limit"] is not None:
            total = min(total, config["limit"])
        mapping = resolve_base(config["dataset"], config["split"])

        registry.update_status(
            eval_id,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total=total,
            base=mapping["base"],
            base_split=mapping["base_split"],
        )

        threshold = config["ambiguous_threshold"]
        confusion = [[0] * num_classes for _ in range(num_classes)]
        correct = failed = ambiguous = 0
        seen = 0
        stopped = False

        loader = DataLoader(
            Subset(dataset, list(range(total))),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                probs = torch.softmax(model(images.to(device)), dim=1).cpu()
                top2 = torch.topk(probs, k=min(2, num_classes), dim=1)

                records = []
                for row in range(labels.size(0)):
                    index = seen + row
                    label = int(labels[row])
                    pred = int(top2.indices[row, 0])
                    conf = float(top2.values[row, 0])
                    margin = conf - (
                        float(top2.values[row, 1]) if num_classes > 1 else 0.0
                    )
                    is_correct = pred == label
                    is_ambiguous = margin < threshold

                    confusion[label][pred] += 1
                    correct += int(is_correct)
                    failed += int(not is_correct)
                    ambiguous += int(is_ambiguous)

                    records.append(
                        {
                            "index": index,
                            "base_index": (
                                mapping["map"][index] if mapping["map"] else index
                            ),
                            "label": label,
                            "pred": pred,
                            "conf": round(conf, 5),
                            "margin": round(margin, 5),
                            "correct": is_correct,
                            "ambiguous": is_ambiguous,
                        }
                    )
                registry.append_results(eval_id, records)
                seen += labels.size(0)

        summary = {
            "total": seen,
            "correct": correct,
            "failed": failed,
            "ambiguous": ambiguous,
            "accuracy": correct / seen if seen else 0.0,
            "ambiguous_threshold": threshold,
        }
        status = registry.update_status(
            eval_id,
            status="stopped" if stopped else "completed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            summary=summary,
            confusion=confusion,
            num_classes=num_classes,
        )
    except Exception as exc:  # noqa: BLE001 - se registra y se re-lanza
        registry.update_status(
            eval_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return status
