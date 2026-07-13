"""Evaluaciones: aplicar una NN a un dataset, filtrar y materializar el filtro."""

from __future__ import annotations

import pytest

from swnist.evaluation.registry import matches
from swnist.evaluation.runner import run_evaluation
from swnist.training.train_features_cnn import run_training
from swnist.validation import ValidationError, validate_filter, validate_train_config


@pytest.fixture
def trained(tiny_config, registry, backups_dir):
    clean = validate_train_config(tiny_config)
    exp_id = registry.create(clean)
    run_training(clean, exp_id, registry, backups_dir=backups_dir)
    return exp_id


@pytest.fixture
def evaluated(trained, registry, evaluations, manager):
    config = manager.validate_evaluation(
        {"experiment": trained, "dataset": "mnist", "split": "test", "limit": 200}
    )
    eval_id = evaluations.create(config)
    run_evaluation(config, eval_id, evaluations, registry)
    return eval_id


# --- runner ----------------------------------------------------------------


def test_la_evaluacion_resume_y_registra_cada_muestra(evaluated, evaluations):
    evaluation = evaluations.get(evaluated)
    summary = evaluation["summary"]

    assert evaluation["status"] == "completed"
    assert summary["total"] == 200
    assert summary["correct"] + summary["failed"] == 200
    assert summary["accuracy"] == pytest.approx(summary["correct"] / 200)

    results = list(evaluations.iter_results(evaluated))
    assert len(results) == 200
    first = results[0]
    assert {"index", "base_index", "label", "pred", "conf", "margin", "correct", "ambiguous"} <= set(first)
    assert first["correct"] == (first["pred"] == first["label"])
    assert first["ambiguous"] == (first["margin"] < summary["ambiguous_threshold"])


def test_la_matriz_de_confusion_cuadra_con_los_resultados(evaluated, evaluations):
    evaluation = evaluations.get(evaluated)
    confusion = evaluation["confusion"]

    assert len(confusion) == 10 and all(len(row) == 10 for row in confusion)
    assert sum(sum(row) for row in confusion) == evaluation["summary"]["total"]

    diagonal = sum(confusion[i][i] for i in range(10))
    assert diagonal == evaluation["summary"]["correct"]

    for result in evaluations.iter_results(evaluated):
        assert confusion[result["label"]][result["pred"]] > 0


def test_filtrar_por_resultado_y_por_clase(evaluated, evaluations):
    todos = evaluations.results(evaluated, filtro={"outcome": "all"}, limit=1000)
    fallos = evaluations.results(evaluated, filtro={"outcome": "failed"}, limit=1000)
    aciertos = evaluations.results(evaluated, filtro={"outcome": "correct"}, limit=1000)
    ambiguas = evaluations.results(evaluated, filtro={"outcome": "ambiguous"}, limit=1000)

    assert todos["total"] == 200
    assert fallos["total"] + aciertos["total"] == 200
    assert all(not r["correct"] for r in fallos["results"])
    assert all(r["correct"] for r in aciertos["results"])
    assert all(r["ambiguous"] for r in ambiguas["results"])

    sietes = evaluations.results(
        evaluated, filtro={"outcome": "all", "label": 7}, limit=1000
    )
    assert sietes["total"] > 0
    assert all(r["label"] == 7 for r in sietes["results"])

    # combinado: los que son 7 y se predijeron como 7
    combinado = evaluations.results(
        evaluated, filtro={"outcome": "all", "label": 7, "pred": 7}, limit=1000
    )
    assert all(r["label"] == 7 and r["pred"] == 7 for r in combinado["results"])
    assert combinado["total"] <= sietes["total"]


def test_paginacion_de_resultados(evaluated, evaluations):
    page = evaluations.results(evaluated, offset=0, limit=10)
    siguiente = evaluations.results(evaluated, offset=10, limit=10)
    assert page["total"] == 200
    assert len(page["results"]) == 10
    assert [r["index"] for r in page["results"]] == list(range(10))
    assert [r["index"] for r in siguiente["results"]] == list(range(10, 20))


def test_matches_aplica_los_tres_criterios():
    result = {"label": 3, "pred": 5, "correct": False, "ambiguous": True}
    assert matches(result, {"outcome": "failed"})
    assert matches(result, {"outcome": "ambiguous", "label": 3, "pred": 5})
    assert not matches(result, {"outcome": "correct"})
    assert not matches(result, {"outcome": "all", "label": 4})
    assert not matches(result, {"outcome": "all", "pred": 3})


# --- materializar el filtro como dataset -----------------------------------


def test_guardar_un_filtro_como_dataset(evaluated, evaluations, datasets):
    filtro = {"outcome": "failed", "label": None, "pred": None}
    indices = evaluations.matching_indices(evaluated, filtro)
    esperados = evaluations.results(evaluated, filtro=filtro, limit=1000)["total"]
    assert len(indices) == esperados

    definition = datasets.create(
        "fallos", base="mnist", split="test", indices=indices, origin={"type": "evaluation"}
    )
    assert definition["size"] == esperados

    # el dataset contiene exactamente las muestras que fallaron
    from swnist.data.registry import build_dataset

    subset = build_dataset("fallos", train=True)
    fallos = {
        r["base_index"]: r["label"]
        for r in evaluations.iter_results(evaluated)
        if not r["correct"]
    }
    assert len(subset) == len(fallos)
    for i in range(len(subset)):
        assert subset[i][1] == fallos[subset.base_index(i)]


def test_los_indices_de_un_custom_apuntan_al_base(trained, registry, evaluations, datasets, manager):
    """Evaluar un custom y guardar OTRO filtro devuelve índices del base, no del subconjunto."""
    datasets.create(
        "primeras-50", base="mnist", split="test", indices=list(range(50)), origin={}
    )
    config = manager.validate_evaluation(
        {"experiment": trained, "dataset": "primeras-50", "split": "train"}
    )
    eval_id = evaluations.create(config)
    run_evaluation(config, eval_id, evaluations, registry)

    evaluation = evaluations.get(eval_id)
    assert evaluation["summary"]["total"] == 50
    assert evaluation["base"] == "mnist"
    assert evaluation["base_split"] == "test"

    # 'index' es la posición dentro del subconjunto; 'base_index', la del base.
    results = list(evaluations.iter_results(eval_id))
    assert [r["index"] for r in results] == list(range(50))
    assert [r["base_index"] for r in results] == list(range(50))

    indices = evaluations.matching_indices(eval_id, {"outcome": "correct"})
    assert all(0 <= i < 10_000 for i in indices)


# --- validación ------------------------------------------------------------


def test_evaluar_una_nn_sin_checkpoint_se_rechaza(registry, tiny_config, manager):
    exp_id = registry.create(validate_train_config(tiny_config))
    with pytest.raises(ValidationError, match="checkpoint"):
        manager.validate_evaluation({"experiment": exp_id, "dataset": "mnist"})


def test_evaluar_un_experimento_inexistente_se_rechaza(manager):
    with pytest.raises(ValidationError, match="no existe"):
        manager.validate_evaluation({"experiment": "exp_fantasma", "dataset": "mnist"})


def test_dataset_desconocido_se_rechaza(trained, manager):
    with pytest.raises(ValidationError, match="desconocido"):
        manager.validate_evaluation({"experiment": trained, "dataset": "otro"})


def test_umbral_de_ambiguedad_invalido(trained, manager):
    with pytest.raises(ValidationError, match="ambiguous_threshold"):
        manager.validate_evaluation(
            {"experiment": trained, "dataset": "mnist", "ambiguous_threshold": 0}
        )


def test_split_invalido(trained, manager):
    with pytest.raises(ValidationError, match="split"):
        manager.validate_evaluation(
            {"experiment": trained, "dataset": "mnist", "split": "otro"}
        )


def test_filtro_invalido():
    with pytest.raises(ValidationError, match="outcome"):
        validate_filter({"outcome": "raros"})
    with pytest.raises(ValidationError, match="label"):
        validate_filter({"outcome": "all", "label": 11})


def test_filtro_normaliza_vacios():
    assert validate_filter({}) == {"outcome": "all", "label": None, "pred": None}
    assert validate_filter({"outcome": "failed", "label": "", "pred": "3"}) == {
        "outcome": "failed",
        "label": None,
        "pred": 3,
    }
