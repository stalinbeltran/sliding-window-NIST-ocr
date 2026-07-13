"""Fixtures: registro y backups en directorio temporal, datasets reducidos (regla 11)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from swnist.experiments.registry import ExperimentRegistry
from swnist.nn_registry import default_config
from swnist.webapp import inference
from swnist.webapp.main import create_app
from swnist.webapp.manager import TrainingManager


@pytest.fixture(autouse=True)
def _clear_inference_caches():
    inference.clear_caches()
    yield
    inference.clear_caches()


@pytest.fixture
def registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "experiments")


@pytest.fixture
def backups_dir(tmp_path):
    return tmp_path / "backups"


@pytest.fixture
def manager(registry, backups_dir) -> TrainingManager:
    return TrainingManager(registry, backups_dir=backups_dir)


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
    )
    with TestClient(app) as test_client:
        test_client.registry = app.state.registry
        test_client.manager = app.state.manager
        yield test_client
