"""Catálogo de NNs entrenables, con configuración por defecto para la web app."""

NNS = {
    "dimensionador": {
        "description": "CNN que extrae características de una ventana de la imagen. "
                       "Se pre-entrena clasificando el dígito de origen de la ventana; "
                       "con dataset.params.empty_fraction > 0 aprende además la clase "
                       "'no hay nada en este recuadro' (10).",
        "defaults": {
            "dataset": {"name": "mnist_full",
                        "params": {"window_size": 28, "windows_per_image": 1,
                                   "empty_fraction": 0.0}},
            "model": {"window_size": 28, "feature_dim": 32, "num_classes": 10, "channels": [16, 32]},
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
