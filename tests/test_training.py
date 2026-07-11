"""Tests end-to-end de entrenamiento (datasets reducidos, registro temporal)."""

import pytest
import torch
import torch.nn as nn

from swnist.training.train_dimensionador import run_training as train_dim
from swnist.training.train_secuenciador import run_training as train_seq, sequence_loss
from swnist.webapp.manager import TrainingManager

from conftest import dim_config, seq_config


def _assert_completed(registry, exp_id, backups_root):
    info = registry.get_experiment(exp_id)
    assert info["status"]["status"] == "completed", info["status"]["error"]
    assert info["status"]["final_test"]["acc"] is not None
    metrics = registry.get_metrics(exp_id)
    assert any(m["type"] == "epoch" for m in metrics)
    assert any(m["type"] == "batch" for m in metrics)
    assert (registry.checkpoints_dir(exp_id) / "best.pt").exists()
    assert (registry.checkpoints_dir(exp_id) / "last.pt").exists()
    assert (backups_root / exp_id / "config.json").exists()  # backup automático


@pytest.mark.parametrize("dataset,params", [
    ("mnist_full", {}),  # imagen completa (defaults)
    ("mnist_full", {"window_size": 5, "windows_per_image": 2}),  # adaptado a ventanas
    ("mnist_windows", {"window_size": 14, "windows_per_image": 2}),
])
def test_train_dimensionador(tmp_registry, tiny_datasets, tmp_path, dataset, params):
    cfg = dim_config(dataset, params)
    exp_id = tmp_registry.create_experiment("dimensionador", cfg)
    train_dim(exp_id, cfg, tmp_registry)
    _assert_completed(tmp_registry, exp_id, tmp_path / "backups")


def test_train_secuenciador_end_to_end(tmp_registry, tiny_datasets, tmp_path):
    # 1. dimensionador previo
    cfg_dim = dim_config("mnist_windows", {"window_size": 14, "windows_per_image": 2})
    dim_id = tmp_registry.create_experiment("dimensionador", cfg_dim)
    train_dim(dim_id, cfg_dim, tmp_registry)

    # 2. secuenciador que lo consume
    cfg = seq_config(dim_id)
    seq_id = tmp_registry.create_experiment("secuenciador", cfg)
    train_seq(seq_id, cfg, tmp_registry)
    _assert_completed(tmp_registry, seq_id, tmp_path / "backups")

    # métricas por paso de la secuencia (ventana 14, stride 7 -> 9 pasos)
    epoch = next(m for m in tmp_registry.get_metrics(seq_id) if m["type"] == "epoch")
    assert len(epoch["val_per_step_acc"]) == 9
    assert len(tmp_registry.get_experiment(seq_id)["status"]["final_test"]["per_step_acc"]) == 9


def test_sequence_loss_modes():
    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(4, 9, 10)
    labels = torch.randint(0, 10, (4,))
    losses = {m: sequence_loss(logits, labels, m, criterion) for m in ("final", "all", "ramp")}
    for v in losses.values():
        assert torch.isfinite(v)
    with pytest.raises(ValueError):
        sequence_loss(logits, labels, "otro", criterion)


def test_secuenciador_requires_dimensionador(tmp_registry, tmp_eval_registry, tiny_datasets):
    """Sin dimensionador_experiment se rechaza antes de crear el experimento."""
    manager = TrainingManager(tmp_registry, tmp_eval_registry)
    with pytest.raises(ValueError, match="dimensionador_experiment"):
        manager.start("secuenciador", seq_config(dim_exp_id=None))
    assert tmp_registry.list_experiments() == []


def test_retrain_continues_from_checkpoint(tmp_registry, tiny_datasets, tmp_path):
    """init_from carga los pesos del experimento origen antes de seguir entrenando."""
    import torch
    cfg = dim_config()
    exp_id = tmp_registry.create_experiment("dimensionador", cfg)
    train_dim(exp_id, cfg, tmp_registry)

    cfg2 = dim_config(**{"init_from": exp_id})
    exp2 = tmp_registry.create_experiment("dimensionador", cfg2)
    train_dim(exp2, cfg2, tmp_registry)
    _assert_completed(tmp_registry, exp2, tmp_path / "backups")

    # el último checkpoint del re-entrenamiento parte de pesos del origen (no aleatorios):
    orig = torch.load(tmp_registry.checkpoints_dir(exp_id) / "best.pt", weights_only=False)
    cont = torch.load(tmp_registry.checkpoints_dir(exp2) / "best.pt", weights_only=False)
    assert orig["model_config"] == cont["model_config"]


def test_manager_rejects_unknown_nn(tmp_registry):
    with pytest.raises(ValueError):
        TrainingManager(tmp_registry).start("otra_nn", {})
