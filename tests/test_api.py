"""Tests de la API web (TestClient), incluida la reproducción del bug original."""

import pytest
from fastapi.testclient import TestClient

import swnist.webapp.main as main
from swnist.webapp.manager import TrainingManager

from conftest import dim_config


@pytest.fixture
def client(tmp_registry, tiny_datasets, monkeypatch):
    manager = TrainingManager(tmp_registry)
    monkeypatch.setattr(main, "registry", tmp_registry)
    monkeypatch.setattr(main, "manager", manager)
    return TestClient(main.app), manager


def _wait(manager, exp_id, timeout=300):
    manager.jobs[exp_id]["thread"].join(timeout=timeout)
    assert not manager.jobs[exp_id]["thread"].is_alive(), "el entrenamiento no terminó a tiempo"


def test_api_nns(client):
    c, _ = client
    names = [n["name"] for n in c.get("/api/nns").json()]
    assert names == ["dimensionador", "secuenciador"]


def test_api_datasets_filtered(client):
    c, _ = client
    for d in c.get("/api/datasets", params={"nn": "dimensionador"}).json():
        assert "dimensionador" in d["compatible_with"]
    seq = c.get("/api/datasets", params={"nn": "secuenciador"}).json()
    assert [d["name"] for d in seq] == ["mnist_sliding_sequences"]


def test_api_train_unknown_nn(client):
    c, _ = client
    r = c.post("/api/train", json={"nn": "no_existe", "config": {}})
    assert r.status_code == 400


def test_api_experiment_not_found(client):
    c, _ = client
    assert c.get("/api/experiments/exp_inexistente").status_code == 404


def test_api_original_bug_now_fails_clearly(client):
    """La config exacta que falló en la UI: mnist_full con params de mnist_windows.

    Ya no debe producir un TypeError críptico sino un ValueError explicativo.
    """
    c, manager = client
    cfg = dim_config("mnist_full", {"window_size": 14, "windows_per_image": 4})
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 200
    exp_id = r.json()["experiment_id"]
    _wait(manager, exp_id)
    status = c.get(f"/api/experiments/{exp_id}").json()["status"]
    assert status["status"] == "failed"
    assert "Parámetros no válidos" in status["error"]
    assert "TypeError" not in status["error"]


def test_api_full_training_flow(client):
    """Entrenamiento completo vía API: mnist_full con params correctos."""
    c, manager = client
    r = c.post("/api/train", json={"nn": "dimensionador", "config": dim_config("mnist_full", {})})
    exp_id = r.json()["experiment_id"]
    _wait(manager, exp_id)

    info = c.get(f"/api/experiments/{exp_id}").json()
    assert info["status"]["status"] == "completed", info["status"]["error"]

    metrics = c.get(f"/api/experiments/{exp_id}/metrics").json()
    assert any(m["type"] == "epoch" for m in metrics)

    listing = c.get("/api/experiments").json()
    assert exp_id in [e["id"] for e in listing]

    # detener un experimento ya terminado responde stopped=false
    assert c.post(f"/api/experiments/{exp_id}/stop").json() == {"stopped": False}
