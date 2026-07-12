"""Tests end-to-end de entrenamiento (datasets reducidos, registro temporal)."""

import pytest
import torch
import torch.nn as nn

from swnist.training.train_dimensionador import run_training as train_dim
from swnist.training.train_secuenciador import run_training as train_seq, sequence_loss
from swnist.webapp.manager import TrainingManager

from conftest import dim_config, seq_config


def _assert_completed(registry, exp_id, backups_root):
    info = registry.get_experiment(exp_id)
    assert info["status"]["status"] == "completed", info["status"]["error"]
    assert info["status"]["final_test"]["acc"] is not None
    metrics = registry.get_metrics(exp_id)
    assert any(m["type"] == "epoch" for m in metrics)
    assert any(m["type"] == "batch" for m in metrics)
    assert (registry.checkpoints_dir(exp_id) / "best.pt").exists()
    assert (registry.checkpoints_dir(exp_id) / "last.pt").exists()
    assert (backups_root / exp_id / "config.json").exists()  # backup automático


@pytest.mark.parametrize("dataset,params", [
    ("synthetic_strokes", {}),                                   # defaults (ventana 5)
    ("synthetic_strokes", {"window_size": 7}),                   # otra ventana
    ("synthetic_strokes", {"curved_fraction": 1.0, "broken_fraction": 0.0}),  # solo curvas continuas
])
def test_train_dimensionador(tmp_registry, tiny_datasets, tmp_path, dataset, params):
    cfg = dim_config(dataset, params)
    exp_id = tmp_registry.create_experiment("dimensionador", cfg)
    train_dim(exp_id, cfg, tmp_registry)
    _assert_completed(tmp_registry, exp_id, tmp_path / "backups")


def test_train_secuenciador_end_to_end(tmp_registry, tiny_datasets, tmp_path):
    # 1. dimensionador previo
    cfg_dim = dim_config("synthetic_strokes", {"window_size": 14})
    dim_id = tmp_registry.create_experiment("dimensionador", cfg_dim)
    train_dim(dim_id, cfg_dim, tmp_registry)

    # 2. secuenciador que lo consume
    cfg = seq_config(dim_id)
    seq_id = tmp_registry.create_experiment("secuenciador", cfg)
    train_seq(seq_id, cfg, tmp_registry)
    _assert_completed(tmp_registry, seq_id, tmp_path / "backups")

    # métricas por paso de la secuencia (ventana 14, stride 7 -> 9 pasos)
    epoch = next(m for m in tmp_registry.get_metrics(seq_id) if m["type"] == "epoch")
    assert len(epoch["val_per_step_acc"]) == 9
    assert len(tmp_registry.get_experiment(seq_id)["status"]["final_test"]["per_step_acc"]) == 9


def test_train_secuenciador_contour_end_to_end(tmp_registry, tiny_datasets, tmp_path):
    """El secuenciador entrena igual con la trayectoria que sigue el trazo."""
    cfg_dim = dim_config("synthetic_strokes", {"window_size": 14})
    dim_id = tmp_registry.create_experiment("dimensionador", cfg_dim)
    train_dim(dim_id, cfg_dim, tmp_registry)

    cfg = seq_config(dim_id)
    cfg["dataset"] = {"name": "mnist_contour_sequences", "params": {"num_steps": 8}}
    seq_id = tmp_registry.create_experiment("secuenciador", cfg)
    train_seq(seq_id, cfg, tmp_registry)
    _assert_completed(tmp_registry, seq_id, tmp_path / "backups")

    # accuracy por paso con T = num_steps fijo
    epoch = next(m for m in tmp_registry.get_metrics(seq_id) if m["type"] == "epoch")
    assert len(epoch["val_per_step_acc"]) == 8


def test_sequence_loss_modes():
    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(4, 9, 10)
    labels = torch.randint(0, 10, (4,))
    losses = {m: sequence_loss(logits, labels, m, criterion) for m in ("final", "all", "ramp")}
    for v in losses.values():
        assert torch.isfinite(v)
    with pytest.raises(ValueError):
        sequence_loss(logits, labels, "otro", criterion)


def test_secuenciador_requires_dimensionador(tmp_registry, tmp_eval_registry, tiny_datasets):
    """Sin dimensionador_experiment se rechaza antes de crear el experimento."""
    manager = TrainingManager(tmp_registry, tmp_eval_registry)
    with pytest.raises(ValueError, match="dimensionador_experiment"):
        manager.start("secuenciador", seq_config(dim_exp_id=None))
    assert tmp_registry.list_experiments() == []


def test_secuenciador_rejects_full_image_dimensionador(tmp_registry, tmp_eval_registry):
    """Un dimensionador de ventana 28 colapsaría la secuencia a 1 paso → 400 antes de crear."""
    dim_id = tmp_registry.create_experiment("dimensionador", dim_config("synthetic_strokes", {"window_size": 28}))
    ckpt = tmp_registry.checkpoints_dir(dim_id) / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.touch()  # la validación solo comprueba existencia, no carga pesos

    manager = TrainingManager(tmp_registry, tmp_eval_registry)
    with pytest.raises(ValueError, match="un solo"):
        manager.start("secuenciador", seq_config(dim_id))
    assert [e["id"] for e in tmp_registry.list_experiments()] == [dim_id]


def test_secuenciador_config_records_effective_window(tmp_registry):
    """La config guardada refleja el window_size real (el del dimensionador)."""
    from swnist.validation import validate_train_config

    dim_id = tmp_registry.create_experiment(
        "dimensionador", dim_config("synthetic_strokes", {"window_size": 14}))
    ckpt = tmp_registry.checkpoints_dir(dim_id) / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.touch()

    cfg = seq_config(dim_id)
    cfg["dataset"]["params"]["window_size"] = 5  # valor engañoso del formulario
    final = validate_train_config("secuenciador", cfg, tmp_registry)
    assert final["dataset"]["params"]["window_size"] == 14


def _fake_trained(registry, nn, cfg):
    """Crea un experimento con checkpoint vacío (la validación solo mira existencia).

    La config pasa por validate_train_config, como hace el manager al entrenar: así
    el experimento registra lo mismo que uno real (trayectoria efectiva,
    model.position_encoding…), que es justo lo que luego se valida.
    """
    from swnist.validation import validate_train_config

    cfg = validate_train_config(nn, cfg, registry)
    exp_id = registry.create_experiment(nn, cfg)
    ckpt = registry.checkpoints_dir(exp_id) / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.touch()
    return exp_id


def test_secuenciador_config_records_effective_stride(tmp_registry):
    """Si el formulario omite el stride, queda registrado el efectivo (default del
    dataset): la trayectoria completa debe poder leerse de la config."""
    from swnist.validation import validate_train_config

    dim_id = _fake_trained(tmp_registry, "dimensionador",
                           dim_config("synthetic_strokes", {"window_size": 14}))
    cfg = seq_config(dim_id)
    del cfg["dataset"]["params"]["stride"]
    final = validate_train_config("secuenciador", cfg, tmp_registry)
    assert final["dataset"]["params"] == {"window_size": 14, "stride": 7}


def test_eval_secuenciador_pins_training_stride(tmp_registry):
    """Evaluar un secuenciador con otro stride → error con razón; sin stride se
    fijan los valores efectivos del entrenamiento en la config de la evaluación."""
    from swnist.validation import validate_eval_config

    dim_id = _fake_trained(tmp_registry, "dimensionador",
                           dim_config("synthetic_strokes", {"window_size": 14}))
    seq_id = _fake_trained(tmp_registry, "secuenciador", seq_config(dim_id))  # stride 7

    with pytest.raises(ValueError, match="stride=7"):
        validate_eval_config({"experiment": seq_id, "split": "test",
                              "dataset": {"name": "mnist_sliding_sequences",
                                          "params": {"stride": 4}}}, tmp_registry)

    out = validate_eval_config({"experiment": seq_id, "split": "test",
                                "dataset": {"name": "mnist_sliding_sequences",
                                            "params": {}}}, tmp_registry)
    assert out["dataset"]["params"] == {"window_size": 14, "stride": 7}


def test_secuenciador_config_records_effective_num_steps(tmp_registry):
    """Con el dataset de trazo, num_steps queda fijado en la config igual que el
    stride en el raster (la trayectoria completa debe poder leerse de la config)."""
    from swnist.validation import validate_train_config

    dim_id = _fake_trained(tmp_registry, "dimensionador",
                           dim_config("synthetic_strokes", {"window_size": 14}))
    cfg = seq_config(dim_id)
    cfg["dataset"] = {"name": "mnist_contour_sequences", "params": {}}
    final = validate_train_config("secuenciador", cfg, tmp_registry)
    assert final["dataset"]["params"] == {"window_size": 14, "num_steps": 12}


def test_eval_secuenciador_pins_training_num_steps(tmp_registry):
    """Evaluar con otro num_steps → error con razón; sin él se fija el efectivo."""
    from swnist.validation import validate_eval_config

    dim_id = _fake_trained(tmp_registry, "dimensionador",
                           dim_config("synthetic_strokes", {"window_size": 14}))
    cfg = seq_config(dim_id)
    cfg["dataset"] = {"name": "mnist_contour_sequences", "params": {"num_steps": 8}}
    seq_id = _fake_trained(tmp_registry, "secuenciador", cfg)

    with pytest.raises(ValueError, match="num_steps=8"):
        validate_eval_config({"experiment": seq_id, "split": "test",
                              "dataset": {"name": "mnist_contour_sequences",
                                          "params": {"num_steps": 20}}}, tmp_registry)

    out = validate_eval_config({"experiment": seq_id, "split": "test",
                                "dataset": {"name": "mnist_contour_sequences",
                                            "params": {}}}, tmp_registry)
    assert out["dataset"]["params"] == {"window_size": 14, "num_steps": 8}


def test_eval_secuenciador_rejects_other_trajectory_type(tmp_registry):
    """Un secuenciador entrenado con recorrido raster no se evalúa con el de trazo
    (y viceversa): la trayectoria es entrada de la red."""
    from swnist.validation import validate_eval_config

    dim_id = _fake_trained(tmp_registry, "dimensionador",
                           dim_config("synthetic_strokes", {"window_size": 14}))
    raster_id = _fake_trained(tmp_registry, "secuenciador", seq_config(dim_id))  # stride 7

    with pytest.raises(ValueError, match="otro tipo de recorrido"):
        validate_eval_config({"experiment": raster_id, "split": "test",
                              "dataset": {"name": "mnist_contour_sequences",
                                          "params": {}}}, tmp_registry)

    cfg = seq_config(dim_id)
    cfg["dataset"] = {"name": "mnist_contour_sequences", "params": {"num_steps": 8}}
    contour_id = _fake_trained(tmp_registry, "secuenciador", cfg)
    with pytest.raises(ValueError, match="otro tipo de recorrido"):
        validate_eval_config({"experiment": contour_id, "split": "test",
                              "dataset": {"name": "mnist_sliding_sequences",
                                          "params": {}}}, tmp_registry)


def test_retrain_continues_from_checkpoint(tmp_registry, tiny_datasets, tmp_path):
    """init_from carga los pesos del experimento origen antes de seguir entrenando."""
    import torch
    cfg = dim_config()
    exp_id = tmp_registry.create_experiment("dimensionador", cfg)
    train_dim(exp_id, cfg, tmp_registry)

    cfg2 = dim_config(**{"init_from": exp_id})
    exp2 = tmp_registry.create_experiment("dimensionador", cfg2)
    train_dim(exp2, cfg2, tmp_registry)
    _assert_completed(tmp_registry, exp2, tmp_path / "backups")

    # el último checkpoint del re-entrenamiento parte de pesos del origen (no aleatorios):
    orig = torch.load(tmp_registry.checkpoints_dir(exp_id) / "best.pt", weights_only=False)
    cont = torch.load(tmp_registry.checkpoints_dir(exp2) / "best.pt", weights_only=False)
    assert orig["model_config"] == cont["model_config"]


def test_manager_rejects_unknown_nn(tmp_registry):
    with pytest.raises(ValueError):
        TrainingManager(tmp_registry).start("otra_nn", {})
