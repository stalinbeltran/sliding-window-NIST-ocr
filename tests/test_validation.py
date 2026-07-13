"""Toda restricción se valida ANTES de crear el experimento, con razón (regla 13)."""

from __future__ import annotations

import copy

import pytest

from swnist.nn_registry import default_config
from swnist.validation import ValidationError, validate_train_config


@pytest.fixture
def config():
    return copy.deepcopy(default_config("features_cnn"))


def test_config_por_defecto_es_valida(config):
    clean = validate_train_config(config)
    assert clean["nn"] == "features_cnn"
    assert clean["model"]["num_classes"] == 10
    assert len(clean["model"]["layers"]) == 2


def test_nn_desconocida(config):
    config["nn"] = "inexistente"
    with pytest.raises(ValidationError, match="desconocida"):
        validate_train_config(config)


def test_dataset_desconocido(config):
    config["dataset"]["name"] = "otro"
    with pytest.raises(ValidationError, match="desconocido"):
        validate_train_config(config)


def test_param_ajeno_al_dataset(config):
    config["dataset"]["params"]["window_size"] = 5
    with pytest.raises(ValidationError, match="no acepta"):
        validate_train_config(config)


def test_sin_capas(config):
    config["model"]["layers"] = []
    with pytest.raises(ValidationError, match="al menos 1 capa"):
        validate_train_config(config)


def test_kernels_debe_ser_entero_positivo(config):
    config["model"]["layers"][0]["kernels"] = 0
    with pytest.raises(ValidationError, match="kernels"):
        validate_train_config(config)


def test_activacion_invalida(config):
    config["model"]["layers"][0]["activation"] = "swish"
    with pytest.raises(ValidationError, match="activación"):
        validate_train_config(config)


def test_pool_type_invalido(config):
    config["model"]["layers"][0]["pool"] = {"type": "median", "size": 2}
    with pytest.raises(ValidationError, match="type"):
        validate_train_config(config)


def test_pool_none_es_valido(config):
    config["model"]["layers"][0]["pool"] = None
    clean = validate_train_config(config)
    assert clean["model"]["layers"][0]["pool"] is None


def test_arquitectura_que_agota_el_mapa_se_rechaza_con_razon(config):
    config["model"]["layers"] = [
        {"kernels": 4, "kernel_size": 3, "padding": 0, "pool": {"type": "max", "size": 2}}
        for _ in range(6)
    ]
    with pytest.raises(ValidationError, match="no cabe|pooling"):
        validate_train_config(config)


def test_epochs_invalidos(config):
    config["training"]["epochs"] = 0
    with pytest.raises(ValidationError, match="epochs"):
        validate_train_config(config)


def test_lr_invalido(config):
    config["training"]["lr"] = 0
    with pytest.raises(ValidationError, match="lr"):
        validate_train_config(config)


def test_val_fraction_invalida(config):
    config["training"]["val_fraction"] = 1.0
    with pytest.raises(ValidationError, match="val_fraction"):
        validate_train_config(config)


def test_init_from_inexistente(config):
    config["init_from"] = "exp_no_existe"
    with pytest.raises(ValidationError, match="no existe"):
        validate_train_config(config, source_experiment=None)


def test_init_from_sin_checkpoint(config):
    source = {"config": copy.deepcopy(config), "has_checkpoint": False}
    config["init_from"] = "exp_origen"
    with pytest.raises(ValidationError, match="checkpoint"):
        validate_train_config(config, source_experiment=source)


def test_init_from_toma_la_arquitectura_del_origen(config):
    source_config = copy.deepcopy(config)
    source_config["model"]["layers"] = [
        {"kernels": 3, "kernel_size": 5, "stride": 1, "padding": 2, "activation": "tanh", "pool": None}
    ]
    source = {"config": source_config, "has_checkpoint": True}

    config["init_from"] = "exp_origen"
    config["model"]["layers"] = [{"kernels": 99, "kernel_size": 3}]  # se ignora
    clean = validate_train_config(config, source_experiment=source)

    assert clean["model"]["layers"] == source_config["model"]["layers"]
    assert clean["init_from"] == "exp_origen"
