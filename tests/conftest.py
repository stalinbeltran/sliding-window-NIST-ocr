"""Fixtures compartidas: datasets pequeños y registros/backup en directorio temporal."""

import pytest

import swnist.evaluation.runner as eval_runner
import swnist.experiments.backup as backup_mod
import swnist.training.train_dimensionador as td
import swnist.training.train_secuenciador as ts
import swnist.webapp.inference as inference
from swnist.data.custom import CustomDatasetStore, CustomSubset
from swnist.data.registry import build_dataset as real_build
from swnist.evaluation.registry import EvaluationRegistry
from swnist.experiments.registry import ExperimentRegistry

N_SMALL = 256


@pytest.fixture
def tiny_datasets(monkeypatch):
    """Reduce todos los datasets a N_SMALL muestras para que los tests sean rápidos.

    Usa CustomSubset como wrapper porque expone display_item (miniaturas/predict).
    """

    def small_build(name, params, train, seed):
        ds = real_build(name, params, train, seed)
        return CustomSubset(ds, range(min(N_SMALL, len(ds))))

    monkeypatch.setattr(td, "build_dataset", small_build)
    monkeypatch.setattr(ts, "build_dataset", small_build)
    monkeypatch.setattr(eval_runner, "build_dataset", small_build)
    monkeypatch.setattr(inference, "build_dataset", small_build)
    inference.clear_caches()
    yield
    inference.clear_caches()


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Registro de experimentos y backups en un directorio temporal."""
    monkeypatch.setattr(backup_mod, "BACKUPS_DIR", tmp_path / "backups")
    return ExperimentRegistry(tmp_path / "experiments")


@pytest.fixture
def tmp_eval_registry(tmp_path):
    return EvaluationRegistry(tmp_path / "evaluations")


@pytest.fixture
def tmp_custom_store(tmp_path, monkeypatch):
    """Almacén de datasets custom en directorio temporal."""
    from swnist.data import registry as data_registry
    store = CustomDatasetStore(tmp_path / "custom_datasets")
    monkeypatch.setattr(data_registry, "custom_store", store)
    return store


def dim_config(dataset_name="synthetic_strokes", dataset_params=None, epochs=1, **extra):
    params = dataset_params or {}
    # model.window_size debe reflejar la entrada real (regla de compatibilidad)
    window = params.get("window_size", 5)
    return {
        "dataset": {"name": dataset_name, "params": params},
        "model": {"window_size": window, "hidden_dim": 16, "channels": [8, 16]},
        "training": {"epochs": epochs, "batch_size": 64, "lr": 0.001, "weight_decay": 0.0,
                     "seed": 42, "val_fraction": 0.1, "log_every": 2},
        **extra,
    }


def seq_config(dim_exp_id, epochs=1, loss_mode="all", cell="gru", **extra):
    return {
        "dimensionador_experiment": dim_exp_id,
        "freeze_dimensionador": True,
        "dataset": {"name": "mnist_contour_sequences", "params": {"num_steps": 9}},
        "model": {"hidden_dim": 32, "num_classes": 10, "cell": cell},
        "training": {"epochs": epochs, "batch_size": 64, "lr": 0.001, "weight_decay": 0.0,
                     "seed": 42, "val_fraction": 0.1, "log_every": 2, "loss_mode": loss_mode},
        **extra,
    }
