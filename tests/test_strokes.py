"""Tests de la uniformización del grosor de trazo (stroke_width).

La transformación redibuja el carácter a partir de su esqueleto: un trazo
grueso y uno delgado del mismo carácter deben terminar viéndose iguales, con
el grosor final parametrizable.
"""

import pytest
import torch

from swnist.data.registry import build_dataset, validate_dataset_params
from swnist.data.strokes import StrokeUniformizer, uniform_stroke

_MEAN, _STD = 0.1307, 0.3081


def _normalize(gray: torch.Tensor) -> torch.Tensor:
    return ((gray.float() - _MEAN) / _STD).unsqueeze(0)


def _gray(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze(0) * _STD + _MEAN


def _vertical_line(width: int) -> torch.Tensor:
    """Trazo vertical de `width` px centrado en la columna 14."""
    g = torch.zeros(28, 28)
    left = 14 - width // 2
    g[4:24, left:left + width] = 1.0
    return _normalize(g)


def _row_widths(image: torch.Tensor, rows=range(8, 20)) -> set[int]:
    """Nº de píxeles con tinta fuerte (> 0.5) por fila, en la zona central."""
    ink = _gray(image) > 0.5
    return {int(ink[r].sum()) for r in rows}


class TestUniformStroke:
    def test_thick_and_thin_converge_to_same_shape(self):
        # Un trazo de 5 px y uno de 1 px con el mismo esqueleto deben quedar
        # iguales tras uniformizar (esa es la razón de ser de la función).
        thick = uniform_stroke(_vertical_line(5), 3)
        thin = uniform_stroke(_vertical_line(1), 3)
        assert _row_widths(thick) == _row_widths(thin) == {3}

    @pytest.mark.parametrize("target", [1, 3, 5])
    def test_target_width_is_parametrizable(self, target):
        out = uniform_stroke(_vertical_line(7), target)
        assert _row_widths(out) == {target}

    def test_deterministic(self):
        img = _vertical_line(5)
        assert torch.equal(uniform_stroke(img, 3), uniform_stroke(img, 3))

    def test_empty_image_passthrough(self):
        img = _normalize(torch.zeros(28, 28))
        assert torch.equal(uniform_stroke(img, 3), img)

    def test_width_zero_disables(self):
        img = _vertical_line(5)
        assert uniform_stroke(img, 0) is img

    def test_uniformizer_cache_matches_direct_call(self):
        img = _vertical_line(5)
        u = StrokeUniformizer(3)
        first, second = u(img, 0), u(img, 0)  # la segunda sale de la caché
        assert torch.equal(first, second)
        # uint8 en caché: misma cuantización que el PNG de las miniaturas
        assert torch.allclose(first, uniform_stroke(img, 3), atol=1.5 / 255 / _STD)


class TestStrokeWidthParam:
    def test_zero_reproduces_original_samples(self):
        plain = build_dataset("mnist_full", {}, train=True, seed=42)
        explicit = build_dataset("mnist_full", {"stroke_width": 0}, train=True, seed=42)
        img_a, label_a = plain[0]
        img_b, label_b = explicit[0]
        assert label_a == label_b and torch.equal(img_a, img_b)

    def test_transforms_samples_deterministically(self):
        ds1 = build_dataset("mnist_full", {"stroke_width": 3}, train=True, seed=42)
        ds2 = build_dataset("mnist_full", {"stroke_width": 3}, train=True, seed=42)
        plain = build_dataset("mnist_full", {}, train=True, seed=42)
        img1, _ = ds1[0]
        img2, _ = ds2[0]
        assert torch.equal(img1, img2)
        assert not torch.equal(img1, plain[0][0])
        # display_item también muestra la imagen transformada
        assert torch.equal(ds1.display_item(0)[0], img1)

    def test_sequence_datasets_accept_stroke_width(self):
        for name, extra in [("mnist_sliding_sequences", {"stride": 7}),
                            ("mnist_contour_sequences", {"num_steps": 8})]:
            ds = build_dataset(name, {**extra, "stroke_width": 2}, train=True, seed=42)
            windows, coords, label = ds[0]
            assert windows.shape[0] == ds.sequence_length == coords.shape[0]
            assert len(ds.trajectory(0)) == ds.sequence_length

    def test_contour_trajectory_follows_transformed_image(self):
        # La trayectoria debe salir de la MISMA imagen que las ventanas: con
        # el trazo uniformizado el esqueleto es (casi) el mismo, así que el
        # recorrido debe seguir existiendo y tener num_steps posiciones.
        ds = build_dataset("mnist_contour_sequences",
                           {"num_steps": 12, "stroke_width": 1}, train=True, seed=42)
        traj = ds.trajectory(0)
        assert len(traj) == 12

    @pytest.mark.parametrize("bad", [-1, 1.5, True, 29, "3"])
    def test_invalid_values_rejected_with_reason(self, bad):
        with pytest.raises(ValueError, match="stroke_width"):
            validate_dataset_params("mnist_full", {"stroke_width": bad})

    def test_accepted_by_every_builtin_dataset(self):
        for name in ("mnist_full", "mnist_windows",
                     "mnist_sliding_sequences", "mnist_contour_sequences"):
            validate_dataset_params(name, {"stroke_width": 3})
