"""Tests de los modelos: formas, config y retroalimentación del estado."""

import pytest
import torch

from swnist.descriptors import NUM_DESCRIPTORS, activate
from swnist.models import Dimensionador, Secuenciador
from swnist.models.secuenciador import relative_positions


def test_dimensionador_shapes():
    # forward → descriptores crudos (6); features → los mismos activados (lo que
    # consume el secuenciador). feature_dim queda fijo = nº de descriptores.
    m = Dimensionador(window_size=5, hidden_dim=16, channels=(8, 16))
    x = torch.randn(4, 1, 5, 5)
    assert m(x).shape == (4, NUM_DESCRIPTORS)
    f = m.features(x)
    assert f.shape == (4, NUM_DESCRIPTORS)
    assert m.config["feature_dim"] == NUM_DESCRIPTORS
    # features == activate(forward): interpretable, en rango natural.
    assert torch.allclose(f, activate(m(x)))


def test_dimensionador_accepts_any_window_size():
    # Pooling adaptativo: la misma red procesa ventanas de otro tamaño
    m = Dimensionador(window_size=5, hidden_dim=16, channels=(8, 16))
    assert m.features(torch.randn(2, 1, 14, 14)).shape == (2, NUM_DESCRIPTORS)


@pytest.mark.parametrize("channels", [(8,), (4, 8, 16), (4, 8, 16, 32, 64)])
def test_dimensionador_variable_depth(channels):
    # channels acepta cualquier número de bloques conv (≥1); con muchas capas el
    # MaxPool se omite cuando la ventana ya no da más y el pooling adaptativo
    # mantiene la salida estable.
    m = Dimensionador(window_size=5, hidden_dim=16, channels=channels)
    x = torch.randn(2, 1, 5, 5)
    assert m.features(x).shape == (2, NUM_DESCRIPTORS)
    assert m(x).shape == (2, NUM_DESCRIPTORS)
    assert m.config["channels"] == list(channels)
    m2 = Dimensionador.from_config(m.config)
    assert m2.config == m.config


def test_dimensionador_rejects_empty_channels():
    with pytest.raises(ValueError):
        Dimensionador(window_size=5, channels=())


def test_dimensionador_from_config_roundtrip():
    m = Dimensionador(window_size=10, hidden_dim=8, channels=(4, 8))
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
    feat, delta = torch.randn(1, 16), torch.rand(1, 2)
    h0 = m.init_state(1)
    with torch.no_grad():
        logits1, h1 = m.step(feat, delta, h0)
        logits2, h2 = m.step(feat, delta, h1)  # misma entrada, estado distinto
    assert not torch.equal(h0, h1)
    assert not torch.allclose(logits1, logits2)


def test_relative_positions():
    """Desplazamiento respecto al paso anterior; el primer paso no tiene anterior."""
    pos = torch.tensor([[[0.0, 0.0], [0.2, 0.1], [0.5, 0.1]]])  # (1, 3, 2)
    d = relative_positions(pos)
    assert torch.allclose(d[0, 0], torch.zeros(2))
    assert torch.allclose(d[0, 1], torch.tensor([0.2, 0.1]))
    assert torch.allclose(d[0, 2], torch.tensor([0.3, 0.0]))


def test_secuenciador_is_translation_invariant():
    """La red recibe (dx, dy), no (x, y): trasladar el recorrido entero no cambia
    la salida (el mismo trazo dibujado más a la derecha se ve igual)."""
    m = Secuenciador(feature_dim=7, hidden_dim=16)
    m.eval()
    feats = torch.randn(2, 6, 7)
    pos = torch.rand(2, 6, 2) * 0.5
    with torch.no_grad():
        a, _ = m(feats, pos)
        b, _ = m(feats, pos + 0.3)          # mismo recorrido, trasladado
        c, _ = m(feats, pos.flip(1))        # otro recorrido: sí debe cambiar
    assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(a, c)


def test_secuenciador_rejects_absolute_positions():
    with pytest.raises(ValueError):
        Secuenciador(feature_dim=7, position_encoding="absolute")


def test_secuenciador_from_config_rejects_legacy_checkpoint():
    """Un secuenciador anterior (posición absoluta) tiene los mismos shapes pero su
    entrada significa otra cosa: debe fallar con razón, no cargar en silencio."""
    m = Secuenciador(feature_dim=7, hidden_dim=16)
    assert m.config["position_encoding"] == "delta"
    assert Secuenciador.from_config(m.config).config == m.config
    legacy = {k: v for k, v in m.config.items() if k != "position_encoding"}
    with pytest.raises(ValueError, match="ABSOLUTA"):
        Secuenciador.from_config(legacy)


def test_secuenciador_elman_cell():
    m = Secuenciador(feature_dim=8, hidden_dim=16, cell="elman")
    logits, _ = m(torch.randn(2, 5, 8), torch.rand(2, 5, 2))
    assert logits.shape == (2, 5, 10)


def test_secuenciador_invalid_cell():
    with pytest.raises(ValueError):
        Secuenciador(feature_dim=8, cell="lstm")
