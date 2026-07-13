"""Semillas y captura del entorno, para poder reproducir un experimento desde cero."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de python, numpy y torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seeded_generator(seed: int) -> torch.Generator:
    """Generador de torch para los DataLoaders (orden de lotes reproducible)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def capture_environment() -> dict[str, Any]:
    """Versiones y plataforma, guardadas en environment.json de cada experimento."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
    }


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
