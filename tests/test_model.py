"""La CNN configurable: N capas, kernels por capa, pooling opcional, capa de salida."""

from __future__ import annotations

import pytest
import torch

from swnist.models.features_cnn import FeaturesCNN, LayerSpec, spatial_trace


def layer(kernels=8, kernel_size=3, padding=1, pool=None, activation="relu", stride=1):
    return {
        "kernels": kernels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "activation": activation,
        "pool": pool,
    }


def build(layers, **kwargs):
    return FeaturesCNN.from_config({"layers": layers, **kwargs})


def test_forward_devuelve_logits_por_clase():
    model = build([layer(kernels=4)])
    logits = model(torch.rand(3, 1, 28, 28))
    assert logits.shape == (3, 10)


def test_numero_de_capas_configurable():
    for n in (1, 2, 3, 4):
        model = build([layer(kernels=4) for _ in range(n)])
        maps = model.feature_maps(torch.rand(1, 1, 28, 28))
        assert len(maps) == n
        assert len(model.blocks) == n


def test_kernels_por_capa_configurables():
    model = build([layer(kernels=6), layer(kernels=11, kernel_size=5, padding=2)])
    maps = model.feature_maps(torch.rand(2, 1, 28, 28))
    assert maps[0].shape == (2, 6, 28, 28)   # 6 kernels, sin pooling → 28x28
    assert maps[1].shape == (2, 11, 28, 28)  # 11 kernels 5x5 con padding 2
    kernels = model.kernels()
    assert kernels[0].shape == (6, 1, 3, 3)
    assert kernels[1].shape == (11, 6, 5, 5)


def test_pooling_es_opcional_y_reduce_el_mapa():
    sin_pool = build([layer(kernels=4)])
    con_pool = build([layer(kernels=4, pool={"type": "max", "size": 2})])
    assert sin_pool.feature_maps(torch.rand(1, 1, 28, 28))[0].shape[-2:] == (28, 28)
    assert con_pool.feature_maps(torch.rand(1, 1, 28, 28))[0].shape[-2:] == (14, 14)


def test_avg_pool_y_stride_propio():
    model = build([layer(kernels=2, pool={"type": "avg", "size": 4, "stride": 2})])
    # (28 - 4) // 2 + 1 = 13
    assert model.feature_maps(torch.rand(1, 1, 28, 28))[0].shape[-2:] == (13, 13)


def test_capa_de_salida_configurable():
    model = build([layer(kernels=4)], num_classes=10, hidden_dim=0)
    assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10)
    assert model.head[-1].out_features == 10


def test_activaciones_soportadas():
    for activation in ("relu", "tanh", "sigmoid", "leaky_relu", "none"):
        model = build([layer(kernels=2, activation=activation)])
        assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10)


def test_relu_produce_features_no_negativos():
    model = build([layer(kernels=4, activation="relu")])
    with torch.no_grad():
        maps = model.feature_maps(torch.rand(1, 1, 28, 28))
    assert float(maps[0].min()) >= 0.0


def test_layer_shapes_describe_la_arquitectura():
    model = build(
        [
            layer(kernels=8, pool={"type": "max", "size": 2}),
            layer(kernels=16, pool={"type": "max", "size": 2}),
        ]
    )
    shapes = model.layer_shapes()
    assert [s["out_size"] for s in shapes] == [14, 7]
    assert [s["kernels"] for s in shapes] == [8, 16]
    assert model.feature_dim == 16 * 7 * 7


def test_arquitectura_que_agota_el_mapa_falla_con_razon():
    layers = [LayerSpec.from_dict(layer(kernel_size=28, padding=0)) for _ in range(2)]
    with pytest.raises(ValueError, match="no cabe"):
        spatial_trace(layers, 28)


def test_pooling_que_no_cabe_falla_con_razon():
    layers = [
        LayerSpec.from_dict(layer(kernel_size=28, padding=0, pool={"type": "max", "size": 2}))
    ]
    with pytest.raises(ValueError, match="pooling"):
        spatial_trace(layers, 28)


def test_sin_capas_falla():
    with pytest.raises(ValueError, match="al menos 1 capa"):
        FeaturesCNN([])
