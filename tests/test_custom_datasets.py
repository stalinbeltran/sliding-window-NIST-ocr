"""Tests del almacén de datasets custom y su integración con el catálogo."""

import pytest
import torch

from swnist.data import registry as data_registry
from swnist.data.custom import slugify
from swnist.data.synthstrokes import STROKE_DEFAULTS


BASE = {"name": "synthetic_strokes", "params": {"num_samples": 300, "window_size": 5},
        "split": "test", "seed": 42}


def test_slugify():
    assert slugify("Mi Dataset á/É") == "mi_dataset_a_e"
    assert slugify("ya-valido_9") == "ya-valido_9"


def test_store_crud(tmp_custom_store):
    store = tmp_custom_store
    created = store.create(BASE, [0, 5, 9], name="fallos_prueba", description="d")
    assert created["name"] == "fallos_prueba" and created["count"] == 3

    # nombre duplicado y vacío rechazados con razón
    with pytest.raises(ValueError, match="Ya existe"):
        store.create(BASE, [1], name="fallos_prueba")
    with pytest.raises(ValueError, match="ninguna muestra"):
        store.create(BASE, [], name="vacio")

    copia = store.copy("fallos_prueba")
    assert copia["name"] == "fallos_prueba_copia" and copia["count"] == 3

    renamed = store.rename("fallos_prueba_copia", "otra")
    assert renamed["name"] == "otra"
    assert not store.exists("fallos_prueba_copia")

    store.update_description("otra", "nueva desc")
    assert store.get_summary("otra")["description"] == "nueva desc"

    store.delete("otra")
    assert [d["name"] for d in store.list()] == ["fallos_prueba"]
    with pytest.raises(KeyError):
        store.get("otra")


def test_catalog_integration_and_build(tmp_custom_store):
    tmp_custom_store.create(BASE, [0, 1, 2], name="sub_strokes")

    listing = data_registry.list_datasets("dimensionador")
    entry = next(d for d in listing if d["name"] == "sub_strokes")
    assert entry["custom"] and entry["count"] == 3
    # hereda compatibilidad del base → no aparece para el secuenciador
    assert all(d["name"] != "sub_strokes"
               for d in data_registry.list_datasets("secuenciador"))

    assert data_registry.effective_window_size("sub_strokes", {}) == 5

    # train=True → subconjunto con los índices guardados (mismas muestras que el base)
    ds = data_registry.build_dataset("sub_strokes", {}, train=True, seed=7)
    base = data_registry.build_dataset("synthetic_strokes", BASE["params"],
                                       train=False, seed=BASE["seed"])
    assert len(ds) == 3
    for i, j in enumerate([0, 1, 2]):
        assert torch.equal(ds[i][0], base[j][0]) and torch.equal(ds[i][1], base[j][1])

    # train=False → test completo del base
    ds_test = data_registry.build_dataset("sub_strokes", {}, train=False, seed=7)
    assert len(ds_test) == len(base)

    # params desconocidos rechazados con mensaje claro
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        data_registry.build_dataset("sub_strokes", {"num_steps": 8}, train=True, seed=7)

    # el catálogo expone los params efectivos del base como defaults del custom
    assert entry["defaults"] == {**STROKE_DEFAULTS, **BASE["params"]}

    # todo param de generación define la muestra: cambiarlo invalidaría los índices
    # guardados → rechazado con razón (window_size, curved_fraction, etc.)
    with pytest.raises(ValueError, match="No se puede cambiar"):
        data_registry.build_dataset("sub_strokes", {"curved_fraction": 0.9},
                                    train=True, seed=7)
    with pytest.raises(ValueError, match="No se puede cambiar"):
        data_registry.build_dataset("sub_strokes", {"window_size": 7}, train=True, seed=7)
