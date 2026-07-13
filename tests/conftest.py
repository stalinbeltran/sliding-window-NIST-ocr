"""Fixtures: registros y datasets custom en directorio temporal (regla 11)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from swnist.data import registry as data_registry
from swnist.evaluation.registry import EvaluationRegistry
from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import default_config
from swnist.webapp import inference
from swnist.webapp.main import create_app
from swnist.webapp.manager import JobManager


@pytest.fixture(autouse=True)
def _isolate(tmp_path):
    """Ni el store de datasets custom ni las cachés de inferencia tocan el repo."""
    inference.clear_caches()
    data_registry.configure(tmp_path / "custom_datasets")
    yield
    inference.clear_caches()
    data_registry.configure(None)


@pytest.fixture
def registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "experiments")


@pytest.fixture
def evaluations(tmp_path) -> EvaluationRegistry:
    return EvaluationRegistry(tmp_path / "evaluations")


@pytest.fixture
def datasets(tmp_path):
    return data_registry.store()


@pytest.fixture
def backups_dir(tmp_path):
    return tmp_path / "backups"


@pytest.fixture
def manager(registry, evaluations, backups_dir) -> JobManager:
    return JobManager(registry, evaluations=evaluations, backups_dir=backups_dir)


@pytest.fixture
def tiny_config() -> dict[str, Any]:
    """Config mínima y rápida: 128 muestras, 1 época, 2 capas."""
    config = copy.deepcopy(default_config("features_cnn"))
    config["dataset"]["params"]["limit"] = 128
    config["training"] = {
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.01,
        "val_fraction": 0.25,
        "log_every": 1,
    }
    return config


@pytest.fixture
def client(tmp_path):
    app = create_app(
        experiments_dir=tmp_path / "experiments",
        backups_dir=tmp_path / "backups",
        evaluations_dir=tmp_path / "evaluations",
        custom_datasets_dir=tmp_path / "custom_datasets",
    )
    with TestClient(app) as test_client:
        test_client.registry = app.state.registry
        test_client.evaluations = app.state.evaluations
        test_client.datasets = app.state.datasets
        test_client.manager = app.state.manager
        yield test_client
