"""Entrenamiento e2e, re-entrenamiento (init_from), checkpoints, métricas y backup."""

from __future__ import annotations

import copy

import pytest
import torch

from swnist.training.common import load_checkpoint
from swnist.training.train_features_cnn import run_training
from swnist.validation import ValidationError, validate_train_config


def train(config, registry, backups_dir, label=None):
    clean = validate_train_config(config)
    exp_id = registry.create(clean, label=label)
    status = run_training(clean, exp_id, registry, backups_dir=backups_dir)
    return exp_id, status


def test_entrenamiento_completo(tiny_config, registry, backups_dir):
    exp_id, status = train(tiny_config, registry, backups_dir)

    assert status["status"] == "completed"
    assert status["best"]["val_acc"] >= 0.0
    assert status["final_test"]["acc"] >= 0.0
    assert status["num_params"] > 0

    # métricas por lote y por época (regla 9)
    records = registry.metrics(exp_id)
    kinds = {record["type"] for record in records}
    assert kinds == {"batch", "epoch"}
    epochs = [r for r in records if r["type"] == "epoch"]
    assert len(epochs) == 1
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "seconds"} <= set(epochs[0])

    # checkpoints
    assert registry.checkpoint_path(exp_id, "best.pt").exists()
    assert registry.checkpoint_path(exp_id, "last.pt").exists()

    # backup automático (regla 3)
    backups = list(backups_dir.iterdir())
    assert len(backups) == 1 and backups[0].name.startswith(exp_id)


def test_el_experimento_guarda_config_y_entorno(tiny_config, registry, backups_dir):
    exp_id, _ = train(tiny_config, registry, backups_dir, label="mi cnn")
    experiment = registry.get(exp_id)

    assert experiment["label"] == "mi cnn"
    assert experiment["config"]["model"]["layers"][0]["kernels"] == 8
    assert experiment["config"]["seed"] == 1234
    assert experiment["environment"]["torch"] == torch.__version__


def test_varias_epocas_registran_una_linea_por_epoca(tiny_config, registry, backups_dir):
    tiny_config["training"]["epochs"] = 2
    exp_id, status = train(tiny_config, registry, backups_dir)
    epochs = [r for r in registry.metrics(exp_id) if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == [1, 2]
    assert status["epochs_done"] == 2


def test_arquitectura_alternativa_sin_pooling(tiny_config, registry, backups_dir):
    tiny_config["model"]["layers"] = [
        {"kernels": 4, "kernel_size": 5, "stride": 1, "padding": 0, "activation": "tanh", "pool": None},
        {"kernels": 6, "kernel_size": 3, "stride": 2, "padding": 0, "activation": "relu", "pool": None},
    ]
    exp_id, status = train(tiny_config, registry, backups_dir)
    assert status["status"] == "completed"
    assert [s["kernels"] for s in status["layer_shapes"]] == [4, 6]


def test_reentrenamiento_desde_un_experimento(tiny_config, registry, backups_dir):
    base_id, _ = train(tiny_config, registry, backups_dir)
    base_weights = load_checkpoint(registry.checkpoint_path(base_id))["model_state"]

    retrain_cfg = copy.deepcopy(tiny_config)
    retrain_cfg["init_from"] = base_id
    retrain_cfg["training"]["lr"] = 0.02
    clean = validate_train_config(retrain_cfg, source_experiment=registry.get(base_id))
    new_id = registry.create(clean)
    status = run_training(clean, new_id, registry, backups_dir=backups_dir)

    assert status["status"] == "completed"
    assert registry.get(new_id)["config"]["init_from"] == base_id

    # Partió de los pesos del origen y siguió entrenando → pesos distintos, misma forma.
    new_weights = load_checkpoint(registry.checkpoint_path(new_id))["model_state"]
    assert set(new_weights) == set(base_weights)
    for key in base_weights:
        assert new_weights[key].shape == base_weights[key].shape
    assert any(
        not torch.allclose(new_weights[key], base_weights[key]) for key in base_weights
    )


def test_reentrenar_usa_la_arquitectura_del_origen(tiny_config, registry, backups_dir):
    base_id, _ = train(tiny_config, registry, backups_dir)

    retrain_cfg = copy.deepcopy(tiny_config)
    retrain_cfg["init_from"] = base_id
    retrain_cfg["model"]["layers"] = [{"kernels": 99, "kernel_size": 7}]  # incompatible
    clean = validate_train_config(retrain_cfg, source_experiment=registry.get(base_id))

    assert clean["model"] == registry.get(base_id)["config"]["model"]
    status = run_training(clean, registry.create(clean), registry, backups_dir=backups_dir)
    assert status["status"] == "completed"  # los pesos cargan porque la arquitectura coincide


def test_reentrenar_desde_experimento_sin_checkpoint_se_rechaza(tiny_config, registry):
    clean = validate_train_config(tiny_config)
    exp_id = registry.create(clean)  # creado pero nunca entrenado

    retrain_cfg = copy.deepcopy(tiny_config)
    retrain_cfg["init_from"] = exp_id
    with pytest.raises(ValidationError, match="checkpoint"):
        validate_train_config(retrain_cfg, source_experiment=registry.get(exp_id))


def test_reproducibilidad_misma_semilla_mismos_pesos(tiny_config, registry, backups_dir):
    first_id, _ = train(copy.deepcopy(tiny_config), registry, backups_dir)
    second_id, _ = train(copy.deepcopy(tiny_config), registry, backups_dir)

    first = load_checkpoint(registry.checkpoint_path(first_id))["model_state"]
    second = load_checkpoint(registry.checkpoint_path(second_id))["model_state"]
    for key in first:
        assert torch.allclose(first[key], second[key]), key


def test_fallo_durante_el_entrenamiento_queda_registrado(tiny_config, registry, backups_dir):
    clean = validate_train_config(tiny_config)
    clean["dataset"]["name"] = "inexistente"  # se salta la validación a propósito
    exp_id = registry.create(clean)

    with pytest.raises(KeyError):
        run_training(clean, exp_id, registry, backups_dir=backups_dir)

    experiment = registry.get(exp_id)
    assert experiment["status"] == "failed"
    assert "inexistente" in experiment["error"]
