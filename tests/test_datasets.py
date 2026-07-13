"""Dataset MNIST y su registro."""

from __future__ import annotations

import pytest
import torch

from swnist.data.registry import build_dataset, effective_params, list_datasets


def test_catalogo_declara_compatibilidad_con_la_nn():
    datasets = list_datasets("features_cnn")
    assert [ds["name"] for ds in datasets] == ["mnist"]
    assert list_datasets("otra_nn") == []


def test_mnist_entrega_imagenes_28x28_y_etiqueta():
    dataset = build_dataset("mnist", train=True, limit=16)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 28, 28)
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert 0 <= label <= 9


def test_limit_recorta_el_split():
    assert len(build_dataset("mnist", train=True, limit=32)) == 32
    assert len(build_dataset("mnist", train=False, limit=10)) == 10
    assert len(build_dataset("mnist", train=False)) == 10_000


def test_display_item_para_las_miniaturas():
    dataset = build_dataset("mnist", train=False, limit=4)
    image = dataset.display_item(0)
    assert image.shape == (28, 28)
    assert image.max() <= 1.0


def test_params_ajenos_se_rechazan_con_razon():
    with pytest.raises(ValueError, match="no acepta"):
        build_dataset("mnist", train=True, stride=2)


def test_dataset_desconocido():
    with pytest.raises(KeyError):
        build_dataset("no_existe", train=True)


def test_effective_params_mezcla_sobre_defaults():
    assert effective_params("mnist", {"limit": 5}) == {"limit": 5}
    assert effective_params("mnist", None) == {"limit": None}
