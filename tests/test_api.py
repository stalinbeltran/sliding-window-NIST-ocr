"""Tests de la API web (TestClient), incluido el flujo completo de pruebas:
evaluación, miniaturas, filtros, datasets custom y CRUD de experimentos."""

import pytest
from fastapi.testclient import TestClient

import swnist.webapp.main as main
from swnist.webapp.manager import TrainingManager

from conftest import dim_config, seq_config


@pytest.fixture
def client(tmp_registry, tmp_eval_registry, tmp_custom_store, tiny_datasets, monkeypatch):
    manager = TrainingManager(tmp_registry, tmp_eval_registry)
    monkeypatch.setattr(main, "registry", tmp_registry)
    monkeypatch.setattr(main, "eval_registry", tmp_eval_registry)
    monkeypatch.setattr(main, "manager", manager)
    return TestClient(main.app), manager


def _wait(manager, job_id, timeout=300):
    manager.jobs[job_id]["thread"].join(timeout=timeout)
    assert not manager.jobs[job_id]["thread"].is_alive(), "el trabajo no terminó a tiempo"


def _train(c, manager, nn="dimensionador", cfg=None):
    r = c.post("/api/train", json={"nn": nn, "config": cfg or dim_config()})
    assert r.status_code == 200, r.json()
    exp_id = r.json()["experiment_id"]
    _wait(manager, exp_id)
    info = c.get(f"/api/experiments/{exp_id}").json()
    assert info["status"]["status"] == "completed", info["status"]["error"]
    return exp_id


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


def test_api_foreign_params_rejected_upfront(client):
    """Params ajenos al dataset se rechazan ANTES de crear el experimento, con la
    razón en el 400. (El caso del bug original — mnist_full con window_size y
    windows_per_image — ahora es válido: mnist_full acepta esos params.)"""
    c, _ = client
    cfg = dim_config("mnist_full", {})
    cfg["dataset"]["params"] = {"stride": 7}
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 400
    assert "Parámetros no válidos" in r.json()["detail"]


def test_api_window_mismatch_rejected_with_reason(client):
    """Entrenar con model.window_size distinto a la entrada del dataset → 400 claro."""
    c, _ = client
    cfg = dim_config("mnist_full", {})
    cfg["model"]["window_size"] = 10  # entradas 28×28 con modelo declarado 10×10
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 400
    assert "Medidas incompatibles" in r.json()["detail"]


def test_api_invalid_channels_rejected_with_reason(client):
    """channels vacío o con valores no positivos → 400 antes de crear el experimento."""
    c, _ = client
    cfg = dim_config("mnist_full", {})
    cfg["model"]["channels"] = []
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 400
    assert "model.channels" in r.json()["detail"]


def test_api_full_training_flow(client):
    """Entrenamiento completo vía API: mnist_full con params correctos."""
    c, manager = client
    exp_id = _train(c, manager, cfg=dim_config("mnist_full", {}))

    metrics = c.get(f"/api/experiments/{exp_id}/metrics").json()
    assert any(m["type"] == "epoch" for m in metrics)

    listing = c.get("/api/experiments").json()
    assert exp_id in [e["id"] for e in listing]

    # detener un experimento ya terminado responde stopped=false
    assert c.post(f"/api/experiments/{exp_id}/stop").json() == {"stopped": False}


def test_api_retrain_init_from(client):
    """Re-entrenar un experimento: init_from carga sus pesos y fuerza su arquitectura."""
    c, manager = client
    exp_id = _train(c, manager)

    cfg = dim_config(**{"init_from": exp_id})
    cfg["model"]["feature_dim"] = 999  # debe ser ignorado: manda el checkpoint origen
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 200
    new_id = r.json()["experiment_id"]
    _wait(manager, new_id)
    info = c.get(f"/api/experiments/{new_id}").json()
    assert info["status"]["status"] == "completed", info["status"]["error"]
    assert info["config"]["init_from"] == exp_id
    assert info["config"]["model"]["feature_dim"] == 16  # la del origen

    # init_from de otra NN → 400 con razón
    cfg2 = seq_config(exp_id, **{"init_from": exp_id})
    r2 = c.post("/api/train", json={"nn": "secuenciador", "config": cfg2})
    assert r2.status_code == 400
    assert "misma arquitectura" in r2.json()["detail"]


def test_api_experiments_crud(client):
    c, manager = client
    exp_id = _train(c, manager)

    # renombrar (alias)
    r = c.patch(f"/api/experiments/{exp_id}", json={"label": "mi dimensionador"})
    assert r.status_code == 200
    assert c.get(f"/api/experiments/{exp_id}").json()["status"]["label"] == "mi dimensionador"

    # duplicar
    copy_id = c.post(f"/api/experiments/{exp_id}/copy").json()["experiment_id"]
    assert copy_id != exp_id
    assert c.get(f"/api/experiments/{copy_id}").json()["status"]["label"] == f"copia de {exp_id}"

    # un secuenciador que referencia al dimensionador bloquea su borrado
    r = c.post("/api/train", json={"nn": "secuenciador", "config": seq_config(exp_id)})
    assert r.status_code == 200, r.json()
    seq_id = r.json()["experiment_id"]
    _wait(manager, seq_id)
    r = c.delete(f"/api/experiments/{exp_id}")
    assert r.status_code == 409
    assert seq_id in r.json()["detail"]

    # eliminando primero el secuenciador, el dimensionador ya se puede borrar
    assert c.delete(f"/api/experiments/{seq_id}").status_code == 200
    assert c.delete(f"/api/experiments/{exp_id}").status_code == 200
    assert c.get(f"/api/experiments/{exp_id}").status_code == 404


def test_api_samples_and_predict(client):
    c, manager = client
    exp_id = _train(c, manager)

    r = c.get("/api/datasets/mnist_windows/samples",
              params={"split": "test", "limit": 5, "params": '{"window_size": 14}'})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0 and len(body["items"]) == 5
    assert {"index", "label", "png"} <= set(body["items"][0])

    # predicción sobre una ventana (sin trace: la muestra ya es una ventana)
    r = c.post("/api/predict", json={
        "experiment": exp_id, "split": "test", "index": 0,
        "dataset": {"name": "mnist_windows", "params": {"window_size": 14}},
    })
    assert r.status_code == 200
    p = r.json()
    assert p["pred"] in range(10) and len(p["probs"]) == 10
    assert p["trace"] is None and p["trace_reason"]

    # sobre imagen completa sí hay trace (deslizamiento del dimensionador)
    r = c.post("/api/predict", json={
        "experiment": exp_id, "split": "test", "index": 0,
        "dataset": {"name": "mnist_full", "params": {}},
    })
    assert r.status_code == 200
    t = r.json()["trace"]
    assert t is not None and t["kind"] == "dimensionador"
    assert len(t["steps"]) == len(t["positions"]) == 9  # ventana 14, stride 7
    assert len(t["steps"][0]["probs"]) == 10


def test_api_dataset_slide(client):
    """GET /slide expone el recorrido real de la ventana para verificar el stride."""
    c, _ = client
    r = c.get("/api/datasets/mnist_sliding_sequences/slide")
    assert r.status_code == 200
    b = r.json()
    assert b["window_size"] == 14 and b["stride"] == 7
    assert b["dataset_uses_stride"] and b["steps"] == 9
    assert b["positions"][0] == [0, 0] and b["axis_coords"] == [0, 7, 14]

    # overrides de visualización: ventana 5, stride 5 → 36 pasos (con borde 23)
    b = c.get("/api/datasets/mnist_sliding_sequences/slide",
              params={"window_size": 5, "stride": 5}).json()
    assert b["steps"] == 36 and b["axis_coords"] == [0, 5, 10, 15, 20, 23]

    # stride > ventana deja franjas sin ver → se avisa (el caso del bug: 25 pasos)
    b = c.get("/api/datasets/mnist_sliding_sequences/slide",
              params={"window_size": 5, "stride": 7}).json()
    assert b["steps"] == 25 and "franjas" in b["note"]

    # dataset sin stride propio (ventanas aleatorias) lo dice claramente
    b = c.get("/api/datasets/mnist_full/slide").json()
    assert not b["dataset_uses_stride"] and "aleatorias" in b["note"]
    assert b["steps"] == 1  # ventana 28 = imagen completa

    assert c.get("/api/datasets/mnist_sliding_sequences/slide",
                 params={"stride": 0}).status_code == 400
    assert c.get("/api/datasets/mnist_sliding_sequences/slide",
                 params={"window_size": 99}).status_code == 400
    assert c.get("/api/datasets/no_existe/slide").status_code == 400


def test_api_secuenciador_eval_and_predict_use_training_stride(client):
    """Bug del 2026-07-11: entrenar con un stride y evaluar/predecir con otro
    cambiaba la trayectoria (entrada de la red) y hundía la accuracy. Ahora la
    evaluación fija los params efectivos del entrenamiento (400 si se pide otro
    stride) y el predict desliza siempre con la trayectoria real."""
    c, manager = client
    dim_id = _train(c, manager)  # dimensionador de ventana 14
    seq_id = _train(c, manager, "secuenciador", seq_config(dim_id))  # stride 7

    # la config del secuenciador registra la trayectoria efectiva
    seq_cfg = c.get(f"/api/experiments/{seq_id}").json()["config"]
    assert seq_cfg["dataset"]["params"] == {"stride": 7, "window_size": 14}

    # evaluar con otro stride → 400 con la razón
    r = c.post("/api/evaluations", json={
        "experiment": seq_id, "split": "test", "limit": 16,
        "dataset": {"name": "mnist_sliding_sequences", "params": {"stride": 4}}})
    assert r.status_code == 400
    assert "stride=7" in r.json()["detail"]

    # sin params → se fijan los efectivos y quedan registrados en la config
    r = c.post("/api/evaluations", json={
        "experiment": seq_id, "split": "test", "limit": 16,
        "dataset": {"name": "mnist_sliding_sequences", "params": {}}})
    assert r.status_code == 200, r.json()
    eval_id = r.json()["evaluation_id"]
    _wait(manager, eval_id)
    info = c.get(f"/api/evaluations/{eval_id}").json()
    assert info["status"]["status"] == "completed", info["status"].get("error")
    assert info["config"]["dataset"]["params"] == {"window_size": 14, "stride": 7}
    # trayectoria de entrenamiento: ventana 14, stride 7 → 9 pasos
    assert len(info["status"]["summary"]["per_step_acc"]) == 9

    # el predict ignora un stride engañoso y usa el del entrenamiento
    r = c.post("/api/predict", json={
        "experiment": seq_id, "split": "test", "index": 0,
        "dataset": {"name": "mnist_sliding_sequences", "params": {"stride": 4}}})
    assert r.status_code == 200, r.json()
    t = r.json()["trace"]
    assert t["kind"] == "secuenciador" and t["stride"] == 7
    assert len(t["steps"]) == len(t["positions"]) == 9


def test_api_evaluation_flow_and_custom_dataset(client, tmp_custom_store):
    c, manager = client
    exp_id = _train(c, manager)

    # dataset incompatible → 400 con razón
    r = c.post("/api/evaluations", json={
        "experiment": exp_id, "dataset": {"name": "mnist_sliding_sequences", "params": {}},
    })
    assert r.status_code == 400
    assert "no es compatible" in r.json()["detail"]

    # evaluación válida
    r = c.post("/api/evaluations", json={
        "experiment": exp_id, "split": "test", "limit": 64,
        "dataset": {"name": "mnist_windows", "params": {"window_size": 14}},
    })
    assert r.status_code == 200, r.json()
    eval_id = r.json()["evaluation_id"]
    _wait(manager, eval_id)

    info = c.get(f"/api/evaluations/{eval_id}").json()
    assert info["status"]["status"] == "completed", info["status"].get("error")
    summary = info["status"]["summary"]
    assert summary["n"] == 64 and 0 <= summary["acc"] <= 1

    # resultados con filtro y miniaturas
    r = c.get(f"/api/evaluations/{eval_id}/results",
              params={"result": "correct", "limit": 10})
    body = r.json()
    assert body["total"] + 0 <= 64
    assert all(it["correct"] for it in body["items"])
    assert all(it.get("png") for it in body["items"])

    # guardar el filtro como dataset custom (nombre por defecto)
    r = c.post(f"/api/evaluations/{eval_id}/save-dataset", json={"result": "correct"})
    assert r.status_code == 200, r.json()
    created = r.json()
    assert created["count"] == body["total"]

    # aparece en el catálogo, compatible con el dimensionador
    listing = c.get("/api/datasets", params={"nn": "dimensionador"}).json()
    entry = next(d for d in listing if d["name"] == created["name"])
    assert entry["custom"] and entry["count"] == created["count"]

    # y se puede usar para entrenar (re-entrenar el mismo experimento)
    cfg = dim_config(created["name"], {}, **{"init_from": exp_id})
    cfg["dataset"]["params"] = {}
    r = c.post("/api/train", json={"nn": "dimensionador", "config": cfg})
    assert r.status_code == 200, r.json()
    _wait(manager, r.json()["experiment_id"])
    info = c.get(f"/api/experiments/{r.json()['experiment_id']}").json()
    assert info["status"]["status"] == "completed", info["status"]["error"]

    # CRUD del dataset custom: renombrar bloqueado mientras lo use un experimento
    r = c.patch(f"/api/custom-datasets/{created['name']}", json={"new_name": "otro"})
    assert r.status_code == 409
    # copiar sí, y el duplicado se puede renombrar y borrar
    copy = c.post(f"/api/custom-datasets/{created['name']}/copy").json()
    r = c.patch(f"/api/custom-datasets/{copy['name']}",
                json={"new_name": "mi_copia", "description": "prueba"})
    assert r.status_code == 200 and r.json()["name"] == "mi_copia"
    assert c.delete("/api/custom-datasets/mi_copia").status_code == 200

    # eliminar la evaluación
    assert c.delete(f"/api/evaluations/{eval_id}").status_code == 200
    assert c.get(f"/api/evaluations/{eval_id}").status_code == 404
