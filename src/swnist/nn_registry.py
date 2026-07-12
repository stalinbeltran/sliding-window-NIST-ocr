"""Catálogo de NNs entrenables, con configuración por defecto para la web app."""

from swnist.data.synthstrokes import STROKE_DEFAULTS

NNS = {
    "dimensionador": {
        "description": "CNN que describe el trazo de una ventana con descriptores "
                       "geométricos interpretables (rectitud, horizontal, vertical, "
                       "ángulo, continuidad). Se entrena sobre el dataset sintético "
                       "de rectas/curvas (synthetic_strokes); esos descriptores son "
                       "lo que consume el secuenciador.",
        "defaults": {
            "dataset": {"name": "synthetic_strokes", "params": dict(STROKE_DEFAULTS)},
            "model": {"window_size": 5, "hidden_dim": 32, "channels": [16, 32]},
            "training": {
                "epochs": 5, "batch_size": 128, "lr": 0.001, "weight_decay": 0.0,
                "seed": 42, "val_fraction": 0.1, "log_every": 50,
            },
        },
    },
    "secuenciador": {
        "description": "Red recurrente: entrada = (x, y) + salida del dimensionador; "
                       "estado interno retroalimentado; salida = dígito reconocido.",
        "defaults": {
            "dimensionador_experiment": None,  # requerido: id de un experimento completado
            "freeze_dimensionador": True,
            "dataset": {"name": "mnist_sliding_sequences", "params": {"stride": 7}},
            "model": {"hidden_dim": 128, "num_classes": 10, "cell": "gru"},
            "training": {
                "epochs": 5, "batch_size": 128, "lr": 0.001, "weight_decay": 0.0,
                "seed": 42, "val_fraction": 0.1, "log_every": 50,
                "loss_mode": "all",  # final | all | ramp
            },
        },
    },
}


def list_nns() -> list[dict]:
    return [{"name": k, "description": v["description"], "defaults": v["defaults"]}
            for k, v in NNS.items()]
