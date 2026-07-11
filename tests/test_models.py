"""Tests de los modelos: formas, config y retroalimentación del estado."""

import pytest
import torch

from swnist.models import Dimensionador, Secuenciador


def test_dimensionador_shapes():
    m = Dimensionador(window_size=14, feature_dim=16, channels=(8, 16))
    x = torch.randn(4, 1, 14, 14)
    assert m.features(x).shape == (4, 16)
    assert m(x).shape == (4, 10)


def test_dimensionador_accepts_any_window_size():
    # Pooling adaptativo: la misma red procesa ventanas de otro tamaño
    m = Dimensionador(window_size=14, feature_dim=16, channels=(8, 16))
    assert m.features(torch.randn(2, 1, 28, 28)).shape == (2, 16)


@pytest.mark.parametrize("channels", [(8,), (4, 8, 16), (4, 8, 16, 32, 64)])
def test_dimensionador_variable_depth(channels):
    # channels acepta cualquier número de bloques conv (≥1); con muchas capas el
    # MaxPool se omite cuando la ventana ya no da más y el pooling adaptativo
    # mantiene la salida estable.
    m = Dimensionador(window_size=14, feature_dim=16, channels=channels)
    x = torch.randn(2, 1, 14, 14)
    assert m.features(x).shape == (2, 16)
    assert m(x).shape == (2, 10)
    assert m.config["channels"] == list(channels)
    m2 = Dimensionador.from_config(m.config)
    assert m2.config == m.config


def test_dimensionador_rejects_empty_channels():
    with pytest.raises(ValueError):
        Dimensionador(window_size=14, channels=())


def test_dimensionador_from_config_roundtrip():
    m = Dimensionador(window_size=10, feature_dim=8, channels=(4, 8))
    m2 = Dimensionador.from_config(m.config)
    assert m2.config == m.config


def test_secuenciador_forward_shapes():
    m = Secuenciador(feature_dim=16, hidden_dim=32)
    feats = torch.randn(4, 9, 16)
    pos = torch.rand(4, 9, 2)
    logits, h = m(feats, pos)
    assert logits.shape == (4, 9, 10)
    assert h.shape == (4, 32)


def test_secuenciador_step_feedback():
    """El estado interno retroalimentado debe cambiar la salida del paso siguiente."""
    m = Secuenciador(feature_dim=16, hidden_dim=32)
    m.eval()
    feat, pos = torch.randn(1, 16), torch.rand(1, 2)
    h0 = m.init_state(1)
    with torch.no_grad():
        logits1, h1 = m.step(feat, pos, h0)
        logits2, h2 = m.step(feat, pos, h1)  # misma entrada, estado distinto
    assert not torch.equal(h0, h1)
    assert not torch.allclose(logits1, logits2)


def test_secuenciador_elman_cell():
    m = Secuenciador(feature_dim=8, hidden_dim=16, cell="elman")
    logits, _ = m(torch.randn(2, 5, 8), torch.rand(2, 5, 2))
    assert logits.shape == (2, 5, 10)


def test_secuenciador_invalid_cell():
    with pytest.raises(ValueError):
        Secuenciador(feature_dim=8, cell="lstm")
