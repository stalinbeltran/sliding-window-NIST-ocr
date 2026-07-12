"""Tests del almacén de datasets custom y su integración con el catálogo."""

import pytest

from swnist.data import registry as data_registry
from swnist.data.custom import slugify


BASE = {"name": "mnist_windows", "params": {"window_size": 14, "windows_per_image": 4},
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
    tmp_custom_store.create(BASE, [0, 1, 2], name="sub_windows")

    listing = data_registry.list_datasets("dimensionador")
    entry = next(d for d in listing if d["name"] == "sub_windows")
    assert entry["custom"] and entry["count"] == 3
    # hereda compatibilidad del base → no aparece para el secuenciador
    assert all(d["name"] != "sub_windows"
               for d in data_registry.list_datasets("secuenciador"))

    assert data_registry.effective_window_size("sub_windows", {}) == 14

    # train=True → subconjunto con los índices guardados (mismas muestras que el base)
    ds = data_registry.build_dataset("sub_windows", {}, train=True, seed=7)
    base = data_registry.build_dataset("mnist_windows", BASE["params"],
                                       train=False, seed=BASE["seed"])
    assert len(ds) == 3
    for i, j in enumerate([0, 1, 2]):
        assert (ds[i][0] == base[j][0]).all() and ds[i][1] == base[j][1]

    # train=False → test completo del base
    ds_test = data_registry.build_dataset("sub_windows", {}, train=False, seed=7)
    assert len(ds_test) == len(base)

    # params desconocidos rechazados con mensaje claro
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        data_registry.build_dataset("sub_windows", {"num_steps": 8}, train=True, seed=7)

    # el catálogo expone los params efectivos del base como defaults del custom
    # (incluidos los que el base no fijó, como empty_fraction, con su default)
    assert entry["defaults"] == {"empty_fraction": 0.0, "stroke_width": 0,
                                 "sampling": "random", "stride": 7,
                                 **BASE["params"]}

    # cambiar windows_per_image invalidaría los índices guardados → rechazado con razón
    with pytest.raises(ValueError, match="windows_per_image"):
        data_registry.build_dataset("sub_windows", {"windows_per_image": 8},
                                    train=True, seed=7)
    # cambiar window_size sí es válido (los índices siguen apuntando a las mismas
    # imágenes/ventanas base): así se re-entrena con otro tamaño de ventana
    ds5 = data_registry.build_dataset("sub_windows", {"window_size": 5}, train=True, seed=7)
    assert ds5[0][0].shape == (1, 5, 5)
