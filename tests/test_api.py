"""API de la web app: catálogos, entrenamiento, CRUD y features."""

from __future__ import annotations

import copy
import time


def train_and_wait(client, config, label=None):
    response = client.post("/api/experiments", json={"config": config, "label": label})
    assert response.status_code == 201, response.text
    exp_id = response.json()["id"]
    client.manager.wait(exp_id, timeout=180)
    return exp_id


def test_catalogo_de_nns(client):
    payload = client.get("/api/nns").json()
    assert [nn["name"] for nn in payload] == ["features_cnn"]
    assert payload[0]["default_config"]["model"]["layers"]


def test_catalogo_de_datasets_filtrado_por_nn(client):
    assert [ds["name"] for ds in client.get("/api/datasets?nn=features_cnn").json()] == ["mnist"]
    assert client.get("/api/datasets?nn=otra").json() == []


def test_config_invalida_responde_400_con_razon(client, tiny_config):
    tiny_config["model"]["layers"][0]["kernels"] = 0
    response = client.post("/api/experiments", json={"config": tiny_config})
    assert response.status_code == 400
    assert "kernels" in response.json()["detail"]


def test_arquitectura_que_agota_el_mapa_responde_400(client, tiny_config):
    tiny_config["model"]["layers"] = [
        {"kernels": 4, "kernel_size": 3, "padding": 0, "pool": {"type": "max", "size": 2}}
        for _ in range(6)
    ]
    response = client.post("/api/experiments", json={"config": tiny_config})
    assert response.status_code == 400
    assert "cabe" in response.json()["detail"]


def test_entrenar_desde_la_api_y_leer_metricas(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config, label="cnn de prueba")

    experiment = client.get(f"/api/experiments/{exp_id}").json()
    assert experiment["status"] == "completed"
    assert experiment["label"] == "cnn de prueba"
    assert experiment["has_checkpoint"] is True
    assert experiment["final_test"]["acc"] >= 0.0

    metrics = client.get(f"/api/experiments/{exp_id}/metrics").json()
    assert metrics["records"]
    assert metrics["next"] == len(metrics["records"])
    assert any(r["type"] == "epoch" for r in metrics["records"])

    # lectura incremental (la usa el polling de la web app)
    tail = client.get(f"/api/experiments/{exp_id}/metrics?since={metrics['next']}").json()
    assert tail["records"] == []


def test_listado_de_experimentos_lo_mas_reciente_primero(client, tiny_config):
    first = train_and_wait(client, tiny_config)
    second = train_and_wait(client, tiny_config)
    ids = [exp["id"] for exp in client.get("/api/experiments").json()]
    assert ids[0] == max(first, second)


def test_reentrenamiento_por_api(client, tiny_config):
    base_id = train_and_wait(client, tiny_config)

    retrain = copy.deepcopy(tiny_config)
    retrain["init_from"] = base_id
    new_id = train_and_wait(client, retrain, label="re-entrenado")

    experiment = client.get(f"/api/experiments/{new_id}").json()
    assert experiment["status"] == "completed"
    assert experiment["config"]["init_from"] == base_id


def test_reentrenar_desde_un_id_inexistente_responde_400(client, tiny_config):
    tiny_config["init_from"] = "exp_no_existe"
    response = client.post("/api/experiments", json={"config": tiny_config})
    assert response.status_code == 400
    assert "no existe" in response.json()["detail"]


def test_crud_de_experimentos(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config)

    renamed = client.patch(f"/api/experiments/{exp_id}", json={"label": "mi cnn"})
    assert renamed.status_code == 200
    assert renamed.json()["label"] == "mi cnn"

    duplicated = client.post(f"/api/experiments/{exp_id}/duplicate")
    assert duplicated.status_code == 201
    copy_id = duplicated.json()["id"]
    assert copy_id != exp_id

    assert client.delete(f"/api/experiments/{copy_id}").status_code == 200
    assert client.get(f"/api/experiments/{copy_id}").status_code == 404


def test_eliminar_bloqueado_si_otro_lo_usa_como_init_from(client, tiny_config):
    base_id = train_and_wait(client, tiny_config)
    child = copy.deepcopy(tiny_config)
    child["init_from"] = base_id
    train_and_wait(client, child)

    response = client.delete(f"/api/experiments/{base_id}")
    assert response.status_code == 409
    assert "init_from" in response.json()["detail"]


def test_experimento_inexistente_responde_404(client):
    assert client.get("/api/experiments/exp_fantasma").status_code == 404


def test_muestras_del_dataset(client):
    payload = client.get("/api/datasets/mnist/samples?split=test&offset=0&limit=6").json()
    assert len(payload["samples"]) == 6
    assert payload["total"] == 10_000
    assert payload["samples"][0]["thumb"].startswith("data:image/png;base64,")


def test_predict_expone_los_features_por_capa(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config)

    response = client.post(
        "/api/predict",
        json={"experiment": exp_id, "dataset": "mnist", "split": "test", "index": 7},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["index"] == 7
    assert 0 <= payload["pred"] <= 9
    assert len(payload["layers"]) == 2
    capa1 = payload["layers"][0]
    assert capa1["kernels"] == 8
    assert len(capa1["maps"]) == 8
    assert len(capa1["maps"][0]["matrix"]) == capa1["height"]


def test_predict_con_indice_invalido_responde_400(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config)
    response = client.post(
        "/api/predict", json={"experiment": exp_id, "index": 10_000_000}
    )
    assert response.status_code == 400
    assert "fuera de rango" in response.json()["detail"]


def test_kernels_aprendidos_por_api(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config)
    payload = client.get(f"/api/experiments/{exp_id}/kernels").json()
    assert len(payload["layers"]) == 2
    assert payload["layers"][0]["kernel_size"] == 3
    assert len(payload["layers"][0]["maps"]) == 8


def test_features_de_una_nn_sin_entrenar_responde_400(client, tiny_config):
    from swnist.validation import validate_train_config

    exp_id = client.registry.create(validate_train_config(tiny_config))
    response = client.get(f"/api/experiments/{exp_id}/kernels")
    assert response.status_code == 400
    assert "checkpoint" in response.json()["detail"]


def test_detener_un_entrenamiento_en_curso(client, tiny_config):
    tiny_config["training"]["epochs"] = 50  # suficiente para alcanzar a detenerlo
    exp_id = client.post("/api/experiments", json={"config": tiny_config}).json()["id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        if client.get(f"/api/experiments/{exp_id}").json()["status"] == "running":
            break
        time.sleep(0.05)

    assert client.post(f"/api/experiments/{exp_id}/stop").status_code == 200
    client.manager.wait(exp_id, timeout=60)

    experiment = client.get(f"/api/experiments/{exp_id}").json()
    assert experiment["status"] == "stopped"
    assert experiment["running"] is False
    # Se detuvo antes de agotar las 50 épocas.
    epochs = [
        r for r in client.get(f"/api/experiments/{exp_id}/metrics").json()["records"]
        if r["type"] == "epoch"
    ]
    assert len(epochs) < 50


def test_detener_un_experimento_que_no_corre_responde_400(client, tiny_config):
    exp_id = train_and_wait(client, tiny_config)
    response = client.post(f"/api/experiments/{exp_id}/stop")
    assert response.status_code == 400
    assert "no está corriendo" in response.json()["detail"]


def test_la_home_sirve_la_web_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Features" in response.text
