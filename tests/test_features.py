"""Extracción de los features reconocidos y su representación matricial."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from swnist.training.train_features_cnn import run_training
from swnist.validation import validate_train_config
from swnist.webapp import inference


@pytest.fixture
def trained(tiny_config, registry, backups_dir):
    """Una CNN entrenada: capa1 = 8 kernels 3x3 + maxpool, capa2 = 16 + maxpool."""
    clean = validate_train_config(tiny_config)
    exp_id = registry.create(clean)
    run_training(clean, exp_id, registry, backups_dir=backups_dir)
    return exp_id


def test_predict_devuelve_prediccion_y_probabilidades(trained, registry):
    payload = inference.predict(trained, registry, dataset="mnist", split="test", index=0)

    assert 0 <= payload["label"] <= 9
    assert 0 <= payload["pred"] <= 9
    assert len(payload["probs"]) == 10
    assert payload["probs"] == pytest.approx(payload["probs"])
    assert sum(payload["probs"]) == pytest.approx(1.0, abs=1e-5)
    assert payload["correct"] == (payload["pred"] == payload["label"])
    assert len(payload["top"]) == 3


def test_los_features_salen_como_matrices_por_capa(trained, registry):
    payload = inference.predict(trained, registry, dataset="mnist", split="test", index=3)
    layers = payload["layers"]

    assert len(layers) == 2  # una entrada por capa convolucional

    capa1, capa2 = layers
    assert capa1["kernels"] == 8 and capa2["kernels"] == 16
    # con maxpool 2x2 en ambas: 28 → 14 → 7
    assert (capa1["height"], capa1["width"]) == (14, 14)
    assert (capa2["height"], capa2["width"]) == (7, 7)

    # un feature map (matriz) por kernel
    assert len(capa1["maps"]) == 8
    assert len(capa2["maps"]) == 16

    mapa = capa1["maps"][0]
    assert len(mapa["matrix"]) == 14
    assert all(len(row) == 14 for row in mapa["matrix"])
    assert mapa["min"] <= mapa["mean"] <= mapa["max"]
    assert all(isinstance(value, float) for row in mapa["matrix"] for value in row)


def test_la_matriz_coincide_con_la_activacion_del_modelo(trained, registry):
    model, _ = inference.get_model(trained, registry)
    data = inference.get_dataset("mnist", train=False)
    image, _ = data[5]

    maps = model.feature_maps(image.unsqueeze(0))
    esperado = maps[0].squeeze(0)[2].detach().numpy()

    payload = inference.predict(trained, registry, dataset="mnist", split="test", index=5)
    obtenido = np.array(payload["layers"][0]["maps"][2]["matrix"])

    assert np.allclose(obtenido, esperado, atol=1e-4)


def test_la_imagen_de_entrada_tambien_va_como_matriz(trained, registry):
    payload = inference.predict(trained, registry, dataset="mnist", split="test", index=1)
    assert len(payload["image"]) == 28
    assert all(len(row) == 28 for row in payload["image"])
    assert payload["thumb"].startswith("data:image/png;base64,")


def test_max_maps_trunca_para_no_reventar_el_navegador(trained, registry):
    payload = inference.predict(
        trained, registry, dataset="mnist", split="test", index=0, max_maps=4
    )
    capa2 = payload["layers"][1]
    assert capa2["kernels"] == 16
    assert capa2["shown"] == 4
    assert capa2["truncated"] is True
    assert len(capa2["maps"]) == 4


def test_kernels_aprendidos_como_matrices(trained, registry):
    payload = inference.kernels(trained, registry)
    capa1 = payload["layers"][0]

    assert capa1["kernels"] == 8
    assert capa1["kernel_size"] == 3
    assert len(capa1["maps"]) == 8
    assert len(capa1["maps"][0]["matrix"]) == 3
    assert len(capa1["maps"][0]["matrix"][0]) == 3


def test_indice_fuera_de_rango_falla_con_razon(trained, registry):
    with pytest.raises(ValueError, match="fuera de rango"):
        inference.predict(trained, registry, dataset="mnist", split="test", index=999_999)


def test_experimento_sin_checkpoint_falla_con_razon(registry, tiny_config):
    exp_id = registry.create(validate_train_config(tiny_config))
    with pytest.raises(ValueError, match="checkpoint"):
        inference.predict(exp_id, registry, dataset="mnist", split="test", index=0)


def test_miniaturas_del_dataset():
    payload = inference.dataset_samples("mnist", split="test", offset=10, limit=5)
    assert payload["total"] == 10_000
    assert payload["offset"] == 10
    assert [s["index"] for s in payload["samples"]] == [10, 11, 12, 13, 14]
    assert all(s["thumb"].startswith("data:image/png;base64,") for s in payload["samples"])


def test_el_png_generado_es_valido():
    image = np.linspace(0, 1, 28 * 28).reshape(28, 28)
    uri = inference.png_data_uri(image)
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in raw[:20] and raw[-8:-4] == b"IEND"
