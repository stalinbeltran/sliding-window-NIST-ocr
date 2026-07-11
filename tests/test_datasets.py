"""Tests de utilidades de ventana, datasets y validación del catálogo."""

import pytest
import torch

from swnist.data.datasets import IMAGE_SIZE, MnistFull, MnistSlidingSequences, MnistWindows
from swnist.data.registry import build_dataset, list_datasets
from swnist.data.windows import extract_window, grid_positions, normalize_position


def test_grid_positions_covers_borders():
    pos = grid_positions(28, 14, 7)
    assert pos[0] == (0, 0)
    assert (14, 14) in pos  # última posición pegada al borde
    assert len(pos) == 9  # coords {0, 7, 14} en cada eje
    # Stride que no divide exacto: igual incluye el borde
    pos2 = grid_positions(28, 14, 9)
    assert max(p[0] for p in pos2) == 14


def test_extract_window_shape():
    img = torch.randn(1, 28, 28)
    w = extract_window(img, 7, 3, 14)
    assert w.shape == (1, 14, 14)
    assert torch.equal(w, img[:, 7:21, 3:17])


def test_normalize_position_range():
    x, y = normalize_position(0, 0, 28, 14)
    assert (x, y) == (0.0, 0.0)
    x, y = normalize_position(14, 14, 28, 14)
    assert (x, y) == (1.0, 1.0)


def test_mnist_full_item():
    ds = MnistFull(train=True)
    img, label = ds[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= label <= 9


def test_mnist_windows_item_and_determinism():
    ds1 = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7)
    ds2 = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7)
    w1, l1 = ds1[10]
    w2, l2 = ds2[10]
    assert w1.shape == (1, 14, 14)
    assert torch.equal(w1, w2) and l1 == l2  # reproducible dado (seed, idx)
    ds3 = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=8)
    w3, _ = ds3[10]
    assert not torch.equal(w1, w3)  # otra semilla, otra ventana


def test_mnist_sliding_sequences_item():
    ds = MnistSlidingSequences(train=True, window_size=14, stride=7)
    windows, positions, label = ds[0]
    T = ds.sequence_length
    assert T == 9
    assert windows.shape == (T, 1, 14, 14)
    assert positions.shape == (T, 2)
    assert positions.min() >= 0 and positions.max() <= 1
    assert 0 <= label <= 9


def test_list_datasets_filters_by_nn():
    for d in list_datasets("dimensionador"):
        assert "dimensionador" in d["compatible_with"]
    seq = [d["name"] for d in list_datasets("secuenciador")]
    assert seq == ["mnist_sliding_sequences"]


def test_build_dataset_unknown_name():
    with pytest.raises(KeyError):
        build_dataset("no_existe", {}, train=True, seed=0)


def test_build_dataset_rejects_params_of_other_dataset():
    # Reproduce el bug original: mnist_full con params de mnist_windows
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_full", {"window_size": 14, "windows_per_image": 4},
                      train=True, seed=42)


def test_build_dataset_valid_params():
    ds = build_dataset("mnist_windows", {"window_size": 10}, train=True, seed=0)
    w, _ = ds[0]
    assert w.shape == (1, 10, 10)
    assert isinstance(build_dataset("mnist_full", {}, train=True, seed=0)[0][0], torch.Tensor)
