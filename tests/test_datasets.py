"""Tests de utilidades de ventana, datasets y validación del catálogo."""

import pytest
import torch

from swnist.data.datasets import (EMPTY_INK_THRESHOLD, EMPTY_LABEL, IMAGE_SIZE,
                                  MnistContourSequences, MnistFull,
                                  MnistSlidingSequences, MnistWindows)
from swnist.data.registry import (build_dataset, effective_window_size,
                                  list_datasets, validate_dataset_params)
from swnist.data.trajectory import (contour_positions, ink_mask, order_path,
                                    resample_path, skeletonize)
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


def test_mnist_full_windowed():
    # mnist_full adaptado a otro tamaño de ventana: muestras = ventanas aleatorias,
    # pero la muestra visible sigue siendo la imagen completa.
    ds = MnistFull(train=True, seed=7, window_size=5, windows_per_image=3)
    assert len(ds) == 3 * len(ds.base)
    w, label = ds[10]
    assert w.shape == (1, 5, 5)
    full, label2 = ds.display_item(10)
    assert full.shape == (1, 28, 28) and label == label2
    # Determinista dado (seed, idx); misma lógica de ventanas que MnistWindows
    ds2 = MnistWindows(train=True, seed=7, window_size=5, windows_per_image=3)
    assert torch.equal(w, ds2[10][0])


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


def test_raster_sampling_windows():
    """sampling='raster': cada imagen produce la grilla completa window_size+stride
    en orden raster (windows_per_image se ignora), sin RNG."""
    ds = MnistWindows(train=True, window_size=14, stride=7, sampling="raster", seed=7)
    pos = grid_positions(IMAGE_SIZE, 14, 7)
    assert len(ds) == len(pos) * len(ds.base)
    image, label = ds.base[0]
    for i, (top, left) in enumerate(pos):
        w, l = ds[i]
        assert l == label
        assert torch.equal(w, extract_window(image, top, left, 14))
    # la segunda imagen empieza donde termina la grilla de la primera
    image2, label2 = ds.base[1]
    w, l = ds[len(pos)]
    assert l == label2 and torch.equal(w, extract_window(image2, 0, 0, 14))
    # windows_per_image se ignora: el número de muestras lo dicta la grilla
    ds2 = MnistWindows(train=True, window_size=14, stride=7, sampling="raster",
                       windows_per_image=99, seed=7)
    assert len(ds2) == len(ds)
    # otra semilla, mismas ventanas: el recorrido es determinista sin RNG
    ds3 = MnistWindows(train=True, window_size=14, stride=7, sampling="raster", seed=8)
    assert torch.equal(ds[3][0], ds3[3][0])


def test_raster_sampling_display_and_empty_fraction():
    # display_item sigue siendo la imagen completa correcta, y empty_fraction
    # funciona igual que con ventanas aleatorias (fracción de ventanas vacías).
    ds = MnistFull(train=True, window_size=14, stride=14, sampling="raster",
                   seed=3, empty_fraction=0.5)
    per_image = len(grid_positions(IMAGE_SIZE, 14, 14))  # coords {0, 14} → 4
    assert per_image == 4 and len(ds) == 4 * len(ds.base)
    full, _ = ds.display_item(per_image)  # primera ventana de la segunda imagen
    assert torch.equal(full, ds.base[1][0])
    n = 200
    labels = [ds.display_item(i)[1] for i in range(n)]
    empties = [i for i in range(n) if labels[i] == EMPTY_LABEL]
    assert 0.3 < len(empties) / n < 0.7
    for i in empties[:10]:
        w, label = ds[i]
        assert label == EMPTY_LABEL
        assert ((w * 0.3081 + 0.1307) <= EMPTY_INK_THRESHOLD + 1e-6).all()


def test_random_sampling_default_unchanged():
    # sampling='random' (default) reproduce las muestras de configs antiguas;
    # el stride es inerte en ese modo.
    old = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7)
    new = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7,
                       sampling="random", stride=3)
    assert len(old) == len(new)
    for i in (0, 10, 99):
        assert torch.equal(old[i][0], new[i][0]) and old[i][1] == new[i][1]


def test_sampling_param_validation():
    with pytest.raises(ValueError, match="sampling debe ser"):
        build_dataset("mnist_windows", {"sampling": "zigzag"}, train=True, seed=0)
    with pytest.raises(ValueError, match="stride debe ser"):
        build_dataset("mnist_full", {"sampling": "raster", "stride": 0},
                      train=True, seed=0)
    # los datasets de secuencias no aceptan sampling (tienen su propio recorrido)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_sliding_sequences", {"sampling": "raster"},
                      train=True, seed=0)
    ds = build_dataset("mnist_windows", {"sampling": "raster", "window_size": 14,
                                         "stride": 7}, train=True, seed=0)
    assert len(ds) == 9 * len(ds.base)


def test_custom_dataset_rejects_sampling_and_raster_grid_changes(tmp_custom_store):
    """El muestreo (y, en raster, la grilla) definen a qué ventana apunta cada
    índice guardado: cambiarlos en un custom se rechaza con razón."""
    tmp_custom_store.create(
        {"name": "mnist_windows",
         "params": {"window_size": 14, "sampling": "raster", "stride": 7},
         "split": "test", "seed": 42},
        [0, 1, 2], "sub_raster")
    validate_dataset_params("sub_raster", {"sampling": "raster", "stride": 7})  # igual: ok
    with pytest.raises(ValueError, match="sampling"):
        validate_dataset_params("sub_raster", {"sampling": "random"})
    with pytest.raises(ValueError, match="stride"):
        validate_dataset_params("sub_raster", {"stride": 5})
    with pytest.raises(ValueError, match="window_size"):
        validate_dataset_params("sub_raster", {"window_size": 10})
    # con muestreo aleatorio la ventana sí se puede cambiar (regla 19), pero
    # pasar a raster remaparía los índices → rechazado
    tmp_custom_store.create(
        {"name": "mnist_windows", "params": {"window_size": 14}, "split": "test",
         "seed": 42},
        [0, 1], "sub_random")
    validate_dataset_params("sub_random", {"window_size": 10})
    with pytest.raises(ValueError, match="sampling"):
        validate_dataset_params("sub_random", {"sampling": "raster"})


def test_empty_fraction_generates_empty_windows():
    """empty_fraction > 0: esa fracción son ventanas SIN píxeles activos con la
    clase 'no hay nada' (EMPTY_LABEL), deterministas dado (seed, idx)."""
    ds = MnistWindows(train=True, window_size=14, windows_per_image=2, seed=7,
                      empty_fraction=0.5)
    n = 200
    labels = [ds.display_item(i)[1] for i in range(n)]
    empties = [i for i in range(n) if labels[i] == EMPTY_LABEL]
    assert 0.3 < len(empties) / n < 0.7  # ~la fracción pedida
    for i in empties[:20]:
        w, label = ds[i]
        assert label == EMPTY_LABEL and w.shape == (1, 14, 14)
        raw = w * 0.3081 + 0.1307  # de-normalizar al gris original
        assert (raw <= EMPTY_INK_THRESHOLD + 1e-6).all()  # sin píxeles activos
    # las demás muestras conservan su dígito
    digit_idx = next(i for i in range(n) if labels[i] != EMPTY_LABEL)
    assert 0 <= ds[digit_idx][1] <= 9
    # determinista: mismas ventanas y etiquetas en otra instancia
    ds2 = MnistWindows(train=True, window_size=14, windows_per_image=2, seed=7,
                       empty_fraction=0.5)
    for i in (empties[0], digit_idx):
        w1, l1 = ds[i]
        w2, l2 = ds2[i]
        assert torch.equal(w1, w2) and l1 == l2


def test_empty_fraction_zero_keeps_old_samples():
    """empty_fraction=0 (default) reproduce exactamente las muestras de las
    configs previas a la clase 'no hay nada' (no consume RNG extra)."""
    old = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7)
    new = MnistWindows(train=True, window_size=14, windows_per_image=4, seed=7,
                       empty_fraction=0.0)
    for i in (0, 10, 99):
        assert torch.equal(old[i][0], new[i][0]) and old[i][1] == new[i][1]


def test_empty_fraction_full_window_is_blank():
    # Con ventana 28 no hay región que muestrear: la muestra vacía es fondo puro.
    ds = MnistFull(train=True, seed=3, empty_fraction=0.9)
    i = next(i for i in range(20) if ds.display_item(i)[1] == EMPTY_LABEL)
    w, label = ds[i]
    assert label == EMPTY_LABEL and w.shape == (1, 28, 28)
    assert ((w * 0.3081 + 0.1307).abs() <= EMPTY_INK_THRESHOLD).all()


def test_empty_fraction_param_validation():
    with pytest.raises(ValueError, match="empty_fraction debe ser"):
        build_dataset("mnist_windows", {"empty_fraction": 1.0}, train=True, seed=0)
    with pytest.raises(ValueError, match="empty_fraction debe ser"):
        build_dataset("mnist_full", {"empty_fraction": -0.1}, train=True, seed=0)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_sliding_sequences", {"empty_fraction": 0.3},
                      train=True, seed=0)
    validate_dataset_params("mnist_windows", {"empty_fraction": 0.3})  # válido


def test_custom_dataset_rejects_empty_fraction_change(tmp_custom_store):
    """Cambiar empty_fraction en un custom re-etiquetaría sus muestras → 400."""
    tmp_custom_store.create(
        {"name": "mnist_windows",
         "params": {"window_size": 14, "empty_fraction": 0.3},
         "split": "test", "seed": 42},
        [0, 1, 2], "con_vacios")
    validate_dataset_params("con_vacios", {"empty_fraction": 0.3})  # igual al base: ok
    validate_dataset_params("con_vacios", {"window_size": 10})      # ventana sí se puede
    with pytest.raises(ValueError, match="empty_fraction"):
        validate_dataset_params("con_vacios", {"empty_fraction": 0.5})


def test_mnist_sliding_sequences_item():
    ds = MnistSlidingSequences(train=True, window_size=14, stride=7)
    windows, positions, label = ds[0]
    T = ds.sequence_length
    assert T == 9
    assert windows.shape == (T, 1, 14, 14)
    assert positions.shape == (T, 2)
    assert positions.min() >= 0 and positions.max() <= 1
    assert 0 <= label <= 9


def _synthetic_stroke() -> torch.Tensor:
    """Imagen normalizada con un trazo vertical de 3 px de grosor (col 13-15)."""
    raw = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
    raw[:, 4:24, 13:16] = 1.0
    return (raw - 0.1307) / 0.3081


def test_skeletonize_thins_stroke():
    mask = ink_mask(_synthetic_stroke())
    skel = skeletonize(mask)
    assert skel.sum() > 0
    assert skel.sum() < mask.sum()  # más delgado que el trazo original
    assert (mask | ~skel).all()     # el esqueleto vive dentro de la tinta
    # ~1 px de grosor: ninguna fila del trazo conserva sus 3 columnas
    assert all(skel[r].sum() <= 2 for r in range(4, 24))


def test_order_path_follows_stroke():
    path = order_path(skeletonize(ink_mask(_synthetic_stroke())))
    rows = [r for r, _ in path]
    # el trazo vertical se recorre de un extremo al otro, sin saltos hacia atrás
    assert rows == sorted(rows) or rows == sorted(rows, reverse=True)


def test_resample_path_fixed_length():
    path = [(0, 0), (10, 0)]
    out = resample_path(path, 6)
    assert len(out) == 6
    assert out[0] == (0, 0) and out[-1] == (10, 0)
    assert resample_path([(3, 3)], 4) == [(3, 3)] * 4
    assert resample_path([], 4) == []


def test_contour_positions_on_ink():
    img = _synthetic_stroke()
    pos = contour_positions(img, window_size=9, num_steps=8)
    assert len(pos) == 8
    mask = ink_mask(img)
    for top, left in pos:
        assert 0 <= top <= IMAGE_SIZE - 9 and 0 <= left <= IMAGE_SIZE - 9
        # cada ventana contiene tinta (la trayectoria sigue el trazo)
        assert mask[top:top + 9, left:left + 9].any()
    # imagen sin tinta: caso degenerado, ventana centrada
    blank = torch.full((1, IMAGE_SIZE, IMAGE_SIZE), (0.0 - 0.1307) / 0.3081)
    assert contour_positions(blank, 9, 4) == [(9, 9)] * 4


def test_mnist_contour_sequences_item():
    ds = MnistContourSequences(train=True, window_size=14, num_steps=10)
    windows, positions, label = ds[0]
    assert ds.sequence_length == 10
    assert windows.shape == (10, 1, 14, 14)
    assert positions.shape == (10, 2)
    assert positions.min() >= 0 and positions.max() <= 1
    assert 0 <= label <= 9
    full, label2 = ds.display_item(0)
    assert full.shape == (1, 28, 28) and label == label2


def test_mnist_contour_sequences_deterministic_and_per_sample():
    ds1 = MnistContourSequences(train=True, window_size=14, num_steps=10)
    ds2 = MnistContourSequences(train=True, window_size=14, num_steps=10)
    w1, p1, _ = ds1[0]
    w2, p2, _ = ds2[0]
    assert torch.equal(w1, w2) and torch.equal(p1, p2)  # determinista (sin RNG)
    assert ds1.trajectory(0) == ds2.trajectory(0)
    # la trayectoria depende del carácter: dos muestras distintas difieren
    assert ds1.trajectory(0) != ds1.trajectory(1)
    # las ventanas de una muestra real contienen tinta en su mayoría
    mask = ink_mask(ds1.base[0][0])
    hits = sum(mask[t:t + 14, l:l + 14].any() for t, l in ds1.trajectory(0))
    assert hits == len(ds1.trajectory(0))


def test_list_datasets_filters_by_nn():
    # Solo los builtin: la lista real puede incluir datasets custom del usuario.
    for d in list_datasets("dimensionador"):
        assert "dimensionador" in d["compatible_with"]
    seq = [d["name"] for d in list_datasets("secuenciador") if not d["custom"]]
    assert seq == ["mnist_sliding_sequences", "mnist_contour_sequences"]


def test_build_dataset_unknown_name():
    with pytest.raises(KeyError):
        build_dataset("no_existe", {}, train=True, seed=0)


def test_build_dataset_rejects_params_of_other_dataset():
    # mnist_full acepta params de ventana, pero no los de otros datasets (p. ej. num_steps)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_full", {"num_steps": 8}, train=True, seed=42)


def test_build_dataset_rejects_invalid_param_values():
    with pytest.raises(ValueError, match="window_size debe ser"):
        build_dataset("mnist_full", {"window_size": 0}, train=True, seed=42)
    with pytest.raises(ValueError, match="window_size debe ser"):
        build_dataset("mnist_windows", {"window_size": 29}, train=True, seed=42)
    with pytest.raises(ValueError, match="windows_per_image debe ser"):
        build_dataset("mnist_full", {"windows_per_image": 0}, train=True, seed=42)
    with pytest.raises(ValueError, match="num_steps debe ser"):
        build_dataset("mnist_contour_sequences", {"num_steps": 1}, train=True, seed=42)
    # cada dataset de secuencias acepta solo su propio parámetro de trayectoria
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_contour_sequences", {"stride": 7}, train=True, seed=42)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_sliding_sequences", {"num_steps": 8}, train=True, seed=42)


def test_build_dataset_valid_params():
    ds = build_dataset("mnist_windows", {"window_size": 10}, train=True, seed=0)
    w, _ = ds[0]
    assert w.shape == (1, 10, 10)
    assert isinstance(build_dataset("mnist_full", {}, train=True, seed=0)[0][0], torch.Tensor)
    # Cualquier dataset se adapta al tamaño de ventana pedido
    ds_full = build_dataset("mnist_full", {"window_size": 5, "windows_per_image": 100},
                            train=True, seed=0)
    assert ds_full[0][0].shape == (1, 5, 5)
    assert len(ds_full) == 100 * len(ds_full.base)


def test_effective_window_size_follows_params():
    assert effective_window_size("mnist_full", {}) == 28
    assert effective_window_size("mnist_full", {"window_size": 5}) == 5
    assert effective_window_size("mnist_windows", {}) == 14
    assert effective_window_size("mnist_sliding_sequences", {"window_size": 7}) == 7
