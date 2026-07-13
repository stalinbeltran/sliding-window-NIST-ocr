"""Datasets custom: subconjuntos guardados (base + split + índices) y su CRUD."""

from __future__ import annotations

import pytest

from swnist.data.registry import build_dataset, dataset_info, list_datasets, store


@pytest.fixture
def subset(datasets):
    return datasets.create(
        "fallos del 7",
        base="mnist",
        split="test",
        indices=[3, 1, 2, 2],  # se ordenan y se guardan tal cual (con repetidos)
        origin={"type": "test"},
    )


def test_crear_guarda_base_split_e_indices(subset):
    assert subset["name"] == "fallos del 7"
    assert subset["base"] == "mnist"
    assert subset["split"] == "test"
    assert subset["indices"] == [1, 2, 2, 3]
    assert subset["size"] == 4


def test_aparece_en_el_catalogo_heredando_la_compatibilidad(subset):
    names = [ds["name"] for ds in list_datasets("features_cnn")]
    assert "mnist" in names and "fallos del 7" in names

    info = dataset_info("fallos del 7")
    assert info["is_custom"] is True
    assert info["nns"] == ["features_cnn"]  # hereda la compatibilidad del base
    assert info["defaults"] == {}  # sus muestras ya están fijadas por índice


def test_train_da_el_subconjunto_y_test_el_base_completo(subset):
    dataset = build_dataset("fallos del 7", train=True)
    assert len(dataset) == 4

    base = build_dataset("mnist", train=False)
    esperado = [base[i][1] for i in [1, 2, 2, 3]]
    assert [dataset[i][1] for i in range(4)] == esperado

    completo = build_dataset("fallos del 7", train=False)
    assert len(completo) == 10_000


def test_las_muestras_son_las_del_base(subset):
    dataset = build_dataset("fallos del 7", train=True)
    base = build_dataset("mnist", train=False)
    imagen, label = dataset[0]
    assert label == base[1][1]
    assert imagen.shape == (1, 28, 28)
    assert dataset.base_index(0) == 1


def test_no_acepta_parametros(subset):
    with pytest.raises(ValueError, match="no acepta parámetros"):
        build_dataset("fallos del 7", train=True, limit=2)


def test_nombre_invalido(datasets):
    with pytest.raises(ValueError, match="nombre de dataset inválido"):
        datasets.create("", base="mnist", split="test", indices=[1], origin={})
    with pytest.raises(ValueError, match="nombre de dataset inválido"):
        datasets.create("con/barra", base="mnist", split="test", indices=[1], origin={})


def test_no_puede_llamarse_como_un_builtin(datasets):
    with pytest.raises(ValueError, match="builtin"):
        datasets.create("mnist", base="mnist", split="test", indices=[1], origin={})


def test_nombre_duplicado(datasets, subset):
    with pytest.raises(ValueError, match="ya existe"):
        datasets.create("fallos del 7", base="mnist", split="test", indices=[9], origin={})


def test_subconjunto_vacio_se_rechaza_con_razon(datasets):
    with pytest.raises(ValueError, match="vacío"):
        datasets.create("vacio", base="mnist", split="test", indices=[], origin={})


def test_renombrar(datasets, subset):
    datasets.rename("fallos del 7", "fallos-7")
    assert not datasets.exists("fallos del 7")
    assert datasets.get("fallos-7")["indices"] == [1, 2, 2, 3]


def test_renombrar_a_uno_existente_falla(datasets, subset):
    datasets.create("otro", base="mnist", split="test", indices=[5], origin={})
    with pytest.raises(ValueError, match="ya existe"):
        datasets.rename("otro", "fallos del 7")


def test_eliminar(datasets, subset):
    datasets.delete("fallos del 7")
    assert not datasets.exists("fallos del 7")
    assert [ds["name"] for ds in list_datasets()] == ["mnist"]


def test_listado_lo_mas_reciente_primero(datasets, subset):
    datasets.create("nuevo", base="mnist", split="train", indices=[0], origin={})
    assert [d["name"] for d in datasets.list()][0] in ("nuevo", "fallos del 7")
    assert len(datasets.list()) == 2


def test_el_store_es_el_configurado_por_los_tests(datasets, tmp_path):
    assert store() is datasets
    assert datasets.base_dir == tmp_path / "custom_datasets"
