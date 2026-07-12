"""Tests de utilidades de ventana, datasets y validación del catálogo."""

import pytest
import torch

from swnist.data.datasets import (IMAGE_SIZE, MnistContourSequences,
                                  MnistSlidingSequences)
from swnist.data.registry import (build_dataset, effective_window_size,
                                  list_datasets, validate_dataset_params)
from swnist.data.synthstrokes import STROKE_DEFAULTS, SyntheticStrokes
from swnist.data.trajectory import (contour_positions, ink_mask, order_path,
                                    resample_path, skeletonize)
from swnist.data.windows import extract_window, grid_positions, normalize_position
from swnist.descriptors import IDX, NUM_DESCRIPTORS, category_index


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


# ---------- dataset sintético de trazos (dimensionador) ----------

_MEAN, _STD = 0.1307, 0.3081


def _gray(w: torch.Tensor) -> torch.Tensor:
    return (w * _STD + _MEAN).clamp(0, 1)


def test_synthetic_strokes_item_shape_and_target():
    ds = SyntheticStrokes(train=True, seed=1, num_samples=50, window_size=5,
                          empty_fraction=0.0)
    assert len(ds) == 50
    w, target, mask = ds[0]
    assert w.shape == (1, 5, 5) and w.dtype == torch.float32
    assert target.shape == (NUM_DESCRIPTORS,) and target.dtype == torch.float32
    assert mask.shape == (NUM_DESCRIPTORS,) and (mask == 1).all()  # trazo: todo supervisado
    # Rangos naturales por canal: sigmoid ∈ [0,1], tanh ∈ [-1,1].
    for name in ("straightness", "horizontal", "vertical", "continuity", "ink"):
        assert 0.0 <= float(target[IDX[name]]) <= 1.0
    for name in ("angle_sin2", "angle_cos2"):
        assert -1.0 <= float(target[IDX[name]]) <= 1.0


def test_synthetic_strokes_never_empty():
    """La generación de trazos nunca produce una ventana vacía (regla del usuario);
    las ventanas vacías son solo las de empty_fraction, aquí desactivado."""
    ds = SyntheticStrokes(train=True, seed=2, num_samples=300, window_size=5,
                          empty_fraction=0.0, length_min=STROKE_DEFAULTS["length_min"])
    for i in range(300):
        w, _, _ = ds[i]
        assert float(_gray(w).max()) > 0.02  # hay tinta


def test_synthetic_strokes_ink_is_coverage():
    """El descriptor ink = cobertura (gris medio) de la ventana."""
    ds = SyntheticStrokes(train=True, seed=4, num_samples=80, window_size=5,
                          empty_fraction=0.0)
    for i in range(80):
        w, t, _ = ds[i]
        assert abs(float(t[IDX["ink"]]) - float(_gray(w).mean())) < 1e-4


def test_synthetic_strokes_empty_windows():
    """empty_fraction: esa fracción son ventanas vacías (ink=0, geometría enmascarada,
    categoría 'vacío'); con empty_fraction=0 nada se enmascara."""
    ds = SyntheticStrokes(train=True, seed=3, num_samples=300, window_size=5,
                          empty_fraction=0.4)
    empties = [i for i in range(300) if float(ds[i][2][IDX["straightness"]]) == 0]
    assert 0.3 < len(empties) / 300 < 0.5
    for i in empties[:20]:
        w, t, m = ds[i]
        assert float(t[IDX["ink"]]) == 0.0 and float(m[IDX["ink"]]) == 1.0
        assert float(m[IDX["straightness"]]) == 0.0  # geometría no supervisada
        assert category_index(t) == 4  # vacío
        assert float(_gray(w).max()) < 1e-5  # ventana en blanco
    ds0 = SyntheticStrokes(train=True, seed=3, num_samples=50, empty_fraction=0.0)
    assert all(float(ds0[i][2].min()) == 1.0 for i in range(50))  # nada enmascarado


def test_synthetic_strokes_deterministic_and_split_disjoint():
    a = SyntheticStrokes(train=True, seed=5, num_samples=20)
    b = SyntheticStrokes(train=True, seed=5, num_samples=20)
    wa, ta, ma = a[7]
    wb, tb, mb = b[7]
    assert torch.equal(wa, wb) and torch.equal(ta, tb) and torch.equal(ma, mb)
    # otra semilla → otra muestra
    c = SyntheticStrokes(train=True, seed=6, num_samples=20)
    assert not torch.equal(a[7][0], c[7][0])
    # train y test son disjuntos (misma seed, distinto split)
    test = SyntheticStrokes(train=False, seed=5, num_samples=20)
    assert not torch.equal(a[7][0], test[7][0])


def test_synthetic_strokes_straight_vs_curved_targets():
    # Solo rectas (sin vacías) → straightness == 1 en todas las muestras.
    straight = SyntheticStrokes(train=True, seed=3, num_samples=40, curved_fraction=0.0,
                                empty_fraction=0.0)
    assert all(float(straight[i][1][IDX["straightness"]]) == 1.0 for i in range(40))
    # Solo curvas → straightness < 1 (hay curvatura).
    curved = SyntheticStrokes(train=True, seed=3, num_samples=40, curved_fraction=1.0,
                              empty_fraction=0.0)
    assert all(float(curved[i][1][IDX["straightness"]]) < 1.0 for i in range(40))


def test_synthetic_strokes_display_item_category():
    ds = SyntheticStrokes(train=True, seed=1, num_samples=10, empty_fraction=0.0)
    w, cat = ds.display_item(0)
    assert w.shape == (1, ds.window_size, ds.window_size)
    assert cat == category_index(ds[0][1]) and 0 <= cat <= 3


def test_synthetic_strokes_via_build_dataset():
    ds = build_dataset("synthetic_strokes", {"num_samples": 30, "window_size": 7},
                       train=True, seed=0)
    w, t, m = ds[0]
    assert w.shape == (1, 7, 7) and t.shape == (NUM_DESCRIPTORS,) and m.shape == (NUM_DESCRIPTORS,)


def test_synthetic_strokes_param_validation():
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("synthetic_strokes", {"stride": 3}, train=True, seed=0)
    with pytest.raises(ValueError, match="Rango inválido"):
        build_dataset("synthetic_strokes", {"angle_min_deg": 100.0, "angle_max_deg": 10.0},
                      train=True, seed=0)
    with pytest.raises(ValueError, match="curved_fraction"):
        build_dataset("synthetic_strokes", {"curved_fraction": 1.5}, train=True, seed=0)
    with pytest.raises(ValueError, match="num_samples"):
        build_dataset("synthetic_strokes", {"num_samples": 0}, train=True, seed=0)
    # rangos válidos (incluye stroke_width como float)
    validate_dataset_params("synthetic_strokes", {"stroke_width": 1.5, "curvature_max": 0.8})


def test_custom_dataset_of_strokes_rejects_param_change(tmp_custom_store):
    """Todo param de generación define la muestra: cambiarlo en un custom se rechaza."""
    tmp_custom_store.create(
        {"name": "synthetic_strokes",
         "params": {"num_samples": 100, "window_size": 5},
         "split": "test", "seed": 42},
        [0, 1, 2], "sub_strokes")
    validate_dataset_params("sub_strokes", {"window_size": 5})  # igual: ok
    with pytest.raises(ValueError, match="No se puede cambiar"):
        validate_dataset_params("sub_strokes", {"curved_fraction": 0.9})


# ---------- secuencias (secuenciador) ----------

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


# ---------- catálogo ----------

def test_list_datasets_filters_by_nn():
    dim = [d["name"] for d in list_datasets("dimensionador") if not d["custom"]]
    assert dim == ["synthetic_strokes"]
    for d in list_datasets("dimensionador"):
        assert "dimensionador" in d["compatible_with"]
    seq = [d["name"] for d in list_datasets("secuenciador") if not d["custom"]]
    assert seq == ["mnist_sliding_sequences", "mnist_contour_sequences"]


def test_build_dataset_unknown_name():
    with pytest.raises(KeyError):
        build_dataset("no_existe", {}, train=True, seed=0)


def test_build_dataset_rejects_params_of_other_dataset():
    # synthetic_strokes no acepta parámetros de secuencias (p. ej. num_steps)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("synthetic_strokes", {"num_steps": 8}, train=True, seed=42)


def test_build_dataset_rejects_invalid_param_values():
    with pytest.raises(ValueError, match="window_size debe ser"):
        build_dataset("mnist_sliding_sequences", {"window_size": 0}, train=True, seed=42)
    with pytest.raises(ValueError, match="num_steps debe ser"):
        build_dataset("mnist_contour_sequences", {"num_steps": 1}, train=True, seed=42)
    # cada dataset de secuencias acepta solo su propio parámetro de trayectoria
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_contour_sequences", {"stride": 7}, train=True, seed=42)
    with pytest.raises(ValueError, match="Parámetros no válidos"):
        build_dataset("mnist_sliding_sequences", {"num_steps": 8}, train=True, seed=42)


def test_effective_window_size_follows_params():
    assert effective_window_size("synthetic_strokes", {}) == 5
    assert effective_window_size("synthetic_strokes", {"window_size": 7}) == 7
    assert effective_window_size("mnist_sliding_sequences", {"window_size": 7}) == 7
