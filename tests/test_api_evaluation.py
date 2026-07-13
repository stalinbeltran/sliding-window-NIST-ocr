"""API de evaluaciones y CRUD de datasets custom."""

from __future__ import annotations

import pytest


def train(client, tiny_config):
    exp_id = client.post("/api/experiments", json={"config": tiny_config}).json()["id"]
    client.manager.wait(exp_id, timeout=180)
    return exp_id


def evaluate(client, exp_id, **overrides):
    config = {"experiment": exp_id, "dataset": "mnist", "split": "test", "limit": 200}
    config.update(overrides)
    response = client.post("/api/evaluations", json={"config": config})
    assert response.status_code == 201, response.text
    eval_id = response.json()["id"]
    client.manager.wait(eval_id, timeout=180)
    return eval_id


@pytest.fixture
def evaluated(client, tiny_config):
    exp_id = train(client, tiny_config)
    return exp_id, evaluate(client, exp_id)


# --- evaluaciones ----------------------------------------------------------


def test_evaluar_por_api(client, evaluated):
    _, eval_id = evaluated
    evaluation = client.get(f"/api/evaluations/{eval_id}").json()

    assert evaluation["status"] == "completed"
    assert evaluation["summary"]["total"] == 200
    assert 0.0 <= evaluation["summary"]["accuracy"] <= 1.0
    assert len(evaluation["confusion"]) == 10


def test_resultados_con_miniaturas_y_filtro(client, evaluated):
    _, eval_id = evaluated
    todos = client.get(f"/api/evaluations/{eval_id}/results?limit=10").json()
    assert todos["total"] == 200
    assert len(todos["results"]) == 10
    assert todos["results"][0]["thumb"].startswith("data:image/png;base64,")

    fallos = client.get(f"/api/evaluations/{eval_id}/results?outcome=failed").json()
    assert all(not r["correct"] for r in fallos["results"])

    sietes = client.get(f"/api/evaluations/{eval_id}/results?label=7").json()
    assert all(r["label"] == 7 for r in sietes["results"])

    confundidos = client.get(
        f"/api/evaluations/{eval_id}/results?outcome=failed&label=7"
    ).json()
    assert all(r["label"] == 7 and not r["correct"] for r in confundidos["results"])


def test_filtro_invalido_responde_400(client, evaluated):
    _, eval_id = evaluated
    assert client.get(f"/api/evaluations/{eval_id}/results?outcome=raros").status_code == 400
    assert client.get(f"/api/evaluations/{eval_id}/results?label=42").status_code == 400


def test_evaluaciones_lo_mas_reciente_primero_y_filtradas_por_experimento(client, tiny_config):
    exp_id = train(client, tiny_config)
    first = evaluate(client, exp_id, limit=50)
    second = evaluate(client, exp_id, limit=50)

    ids = [e["id"] for e in client.get("/api/evaluations").json()]
    assert ids[0] == max(first, second)
    assert client.get(f"/api/evaluations?experiment={exp_id}").json()[0]["id"] == ids[0]
    assert client.get("/api/evaluations?experiment=exp_otro").json() == []


def test_evaluar_una_nn_sin_entrenar_responde_400(client, tiny_config):
    from swnist.validation import validate_train_config

    exp_id = client.registry.create(validate_train_config(tiny_config))
    response = client.post(
        "/api/evaluations", json={"config": {"experiment": exp_id, "dataset": "mnist"}}
    )
    assert response.status_code == 400
    assert "checkpoint" in response.json()["detail"]


def test_crud_de_evaluaciones(client, evaluated):
    _, eval_id = evaluated
    renamed = client.patch(f"/api/evaluations/{eval_id}", json={"label": "prueba 1"})
    assert renamed.json()["label"] == "prueba 1"

    assert client.delete(f"/api/evaluations/{eval_id}").status_code == 200
    assert client.get(f"/api/evaluations/{eval_id}").status_code == 404


def test_no_se_elimina_un_experimento_con_evaluaciones(client, evaluated):
    exp_id, _ = evaluated
    response = client.delete(f"/api/experiments/{exp_id}")
    assert response.status_code == 409
    assert "evaluación" in response.json()["detail"]


# --- datasets desde un filtro ----------------------------------------------


def test_guardar_el_filtro_como_dataset(client, evaluated):
    _, eval_id = evaluated
    fallos = client.get(f"/api/evaluations/{eval_id}/results?outcome=failed&limit=200").json()

    response = client.post(
        "/api/custom-datasets",
        json={
            "name": "fallos-test",
            "from_evaluation": {"evaluation": eval_id, "outcome": "failed"},
        },
    )
    assert response.status_code == 201, response.text
    definition = response.json()
    assert definition["size"] == fallos["total"]
    assert definition["base"] == "mnist"
    assert definition["split"] == "test"
    assert definition["origin"]["evaluation"] == eval_id
    assert definition["origin"]["filter"]["outcome"] == "failed"

    # aparece en el catálogo, compatible con la misma NN
    datasets = client.get("/api/datasets?nn=features_cnn").json()
    assert "fallos-test" in [ds["name"] for ds in datasets]


def test_guardar_filtro_por_clase(client, evaluated):
    _, eval_id = evaluated
    sietes = client.get(f"/api/evaluations/{eval_id}/results?label=7&limit=200").json()
    response = client.post(
        "/api/custom-datasets",
        json={"name": "los-7", "from_evaluation": {"evaluation": eval_id, "label": 7}},
    )
    assert response.status_code == 201
    assert response.json()["size"] == sietes["total"]


def test_un_filtro_sin_coincidencias_se_rechaza_con_razon(client, evaluated):
    _, eval_id = evaluated
    # aciertos etiquetados 3 pero predichos 4: por definición, ninguno
    response = client.post(
        "/api/custom-datasets",
        json={
            "name": "imposible",
            "from_evaluation": {
                "evaluation": eval_id, "outcome": "correct", "label": 3, "pred": 4,
            },
        },
    )
    assert response.status_code == 400
    assert "vacío" in response.json()["detail"]


def test_entrenar_y_re_evaluar_con_el_dataset_guardado(client, evaluated, tiny_config):
    exp_id, eval_id = evaluated
    client.post(
        "/api/custom-datasets",
        json={"name": "fallos", "from_evaluation": {"evaluation": eval_id, "outcome": "failed"}},
    )

    # sirve para entrenar (regla: hereda la compatibilidad del base)
    tiny_config["dataset"] = {"name": "fallos", "params": {}}
    tiny_config["training"]["batch_size"] = 8
    new_id = train(client, tiny_config)
    assert client.get(f"/api/experiments/{new_id}").json()["status"] == "completed"

    # y sirve para volver a evaluar
    re_eval = evaluate(client, exp_id, dataset="fallos", split="train", limit=None)
    evaluation = client.get(f"/api/evaluations/{re_eval}").json()
    assert evaluation["status"] == "completed"
    # el modelo falló en todas ellas, así que su accuracy sobre este subconjunto es 0
    assert evaluation["summary"]["accuracy"] == 0.0


def test_dataset_custom_con_params_en_el_entrenamiento_responde_400(client, evaluated, tiny_config):
    _, eval_id = evaluated
    client.post(
        "/api/custom-datasets",
        json={"name": "fallos", "from_evaluation": {"evaluation": eval_id, "outcome": "failed"}},
    )
    tiny_config["dataset"] = {"name": "fallos", "params": {"limit": 5}}
    response = client.post("/api/experiments", json={"config": tiny_config})
    assert response.status_code == 400
    assert "no acepta parámetros" in response.json()["detail"]


# --- CRUD de datasets -------------------------------------------------------


def test_crear_dataset_recortando_un_base(client):
    response = client.post(
        "/api/custom-datasets",
        json={
            "name": "mnist-test-100",
            "from_base": {"base": "mnist", "split": "test", "limit": 100},
        },
    )
    assert response.status_code == 201, response.text
    definition = response.json()
    assert definition["size"] == 100
    assert definition["origin"] == {
        "type": "base", "base": "mnist", "split": "test", "limit": 100,
    }

    samples = client.get("/api/datasets/mnist-test-100/samples?split=train&limit=5").json()
    assert samples["total"] == 100


def test_listar_leer_renombrar_y_eliminar(client):
    client.post(
        "/api/custom-datasets",
        json={"name": "recorte", "from_base": {"base": "mnist", "split": "train", "limit": 20}},
    )

    listado = client.get("/api/custom-datasets").json()
    assert [d["name"] for d in listado] == ["recorte"]
    assert client.get("/api/custom-datasets/recorte").json()["size"] == 20

    renamed = client.patch("/api/custom-datasets/recorte", json={"name": "recorte-20"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "recorte-20"
    assert client.get("/api/custom-datasets/recorte").status_code == 404

    assert client.delete("/api/custom-datasets/recorte-20").status_code == 200
    assert client.get("/api/custom-datasets").json() == []


def test_no_se_puede_renombrar_ni_eliminar_uno_en_uso(client, evaluated, tiny_config):
    exp_id, eval_id = evaluated
    client.post(
        "/api/custom-datasets",
        json={"name": "en-uso", "from_evaluation": {"evaluation": eval_id, "outcome": "correct"}},
    )
    tiny_config["dataset"] = {"name": "en-uso", "params": {}}
    train(client, tiny_config)  # ahora un experimento lo referencia por nombre

    for response in (
        client.patch("/api/custom-datasets/en-uso", json={"name": "otro"}),
        client.delete("/api/custom-datasets/en-uso"),
    ):
        assert response.status_code == 409
        assert "experimento" in response.json()["detail"]


def test_no_se_puede_llamar_como_un_builtin(client):
    response = client.post(
        "/api/custom-datasets",
        json={"name": "mnist", "from_base": {"base": "mnist", "split": "test", "limit": 10}},
    )
    assert response.status_code == 400
    assert "builtin" in response.json()["detail"]


def test_derivar_de_un_custom_con_from_base_se_rechaza(client):
    client.post(
        "/api/custom-datasets",
        json={"name": "base-custom", "from_base": {"base": "mnist", "split": "test", "limit": 10}},
    )
    response = client.post(
        "/api/custom-datasets",
        json={"name": "derivado", "from_base": {"base": "base-custom", "split": "test"}},
    )
    assert response.status_code == 400
    assert "ya es un dataset custom" in response.json()["detail"]


def test_dataset_inexistente_responde_404(client):
    assert client.get("/api/custom-datasets/fantasma").status_code == 404
    assert client.delete("/api/custom-datasets/fantasma").status_code == 404
