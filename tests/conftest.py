"""Fixtures compartidas: datasets pequeños y registro/backup en directorio temporal."""

import pytest
from torch.utils.data import Subset

import swnist.experiments.backup as backup_mod
import swnist.training.train_dimensionador as td
import swnist.training.train_secuenciador as ts
from swnist.data.registry import build_dataset as real_build
from swnist.experiments.registry import ExperimentRegistry

N_SMALL = 256


@pytest.fixture
def tiny_datasets(monkeypatch):
    """Reduce todos los datasets a N_SMALL muestras para que los tests sean rápidos."""

    def small_build(name, params, train, seed):
        ds = real_build(name, params, train, seed)
        return Subset(ds, range(min(N_SMALL, len(ds))))

    monkeypatch.setattr(td, "build_dataset", small_build)
    monkeypatch.setattr(ts, "build_dataset", small_build)


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Registro de experimentos y backups en un directorio temporal."""
    monkeypatch.setattr(backup_mod, "BACKUPS_DIR", tmp_path / "backups")
    return ExperimentRegistry(tmp_path / "experiments")


def dim_config(dataset_name="mnist_windows", dataset_params=None, epochs=1):
    return {
        "dataset": {"name": dataset_name, "params": dataset_params or {}},
        "model": {"window_size": 14, "feature_dim": 16, "num_classes": 10, "channels": [8, 16]},
        "training": {"epochs": epochs, "batch_size": 64, "lr": 0.001, "weight_decay": 0.0,
                     "seed": 42, "val_fraction": 0.1, "log_every": 2},
    }


def seq_config(dim_exp_id, epochs=1, loss_mode="all", cell="gru"):
    return {
        "dimensionador_experiment": dim_exp_id,
        "freeze_dimensionador": True,
        "dataset": {"name": "mnist_sliding_sequences", "params": {"stride": 7}},
        "model": {"hidden_dim": 32, "num_classes": 10, "cell": cell},
        "training": {"epochs": epochs, "batch_size": 64, "lr": 0.001, "weight_decay": 0.0,
                     "seed": 42, "val_fraction": 0.1, "log_every": 2, "loss_mode": loss_mode},
    }
