"""Registro de experimentos: creación, listado ordenado y CRUD (regla 17)."""

from __future__ import annotations

import pytest

from swnist.validation import validate_train_config


@pytest.fixture
def config(tiny_config):
    return validate_train_config(tiny_config)


def test_crear_deja_los_cuatro_archivos(registry, config):
    exp_id = registry.create(config)
    directory = registry.path(exp_id)
    for name in ("config.json", "environment.json", "status.json", "metrics.jsonl"):
        assert (directory / name).exists()
    assert exp_id.startswith("exp_") and exp_id.endswith("features_cnn")
    assert registry.get(exp_id)["status"] == "created"
    assert registry.get(exp_id)["has_checkpoint"] is False


def test_listado_lo_mas_reciente_primero(registry, config):
    first = registry.create(config)
    second = registry.create(config)
    ids = [exp["id"] for exp in registry.list()]
    assert ids[0] == max(first, second)
    assert set(ids) == {first, second}


def test_renombrar_es_un_alias_el_id_no_cambia(registry, config):
    exp_id = registry.create(config)
    registry.rename(exp_id, "mi red")
    assert registry.get(exp_id)["label"] == "mi red"
    assert registry.get(exp_id)["id"] == exp_id


def test_renombrar_vacio_falla(registry, config):
    exp_id = registry.create(config)
    with pytest.raises(ValueError):
        registry.rename(exp_id, "   ")


def test_duplicar_copia_con_id_nuevo(registry, config):
    exp_id = registry.create(config)
    copy_id = registry.duplicate(exp_id)
    assert copy_id != exp_id
    assert registry.get(copy_id)["config"] == registry.get(exp_id)["config"]
    assert "(copia)" in registry.get(copy_id)["label"]


def test_eliminar(registry, config):
    exp_id = registry.create(config)
    registry.delete(exp_id)
    assert not registry.exists(exp_id)


def test_no_se_elimina_si_otro_experimento_lo_usa_como_init_from(registry, config, tiny_config):
    base_id = registry.create(config)
    child_cfg = dict(config)
    child_cfg["init_from"] = base_id
    registry.create(child_cfg)

    with pytest.raises(ValueError, match="init_from"):
        registry.delete(base_id)


def test_metricas_se_leen_incrementalmente(registry, config):
    exp_id = registry.create(config)
    registry.log_metric(exp_id, {"type": "batch", "step": 1, "loss": 2.3})
    registry.log_metric(exp_id, {"type": "batch", "step": 2, "loss": 1.9})
    assert len(registry.metrics(exp_id)) == 2
    assert registry.metrics(exp_id, since=1)[0]["step"] == 2
