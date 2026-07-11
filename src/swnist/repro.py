"""Reproducibility helpers: seeding and environment capture."""

import platform
import random
import subprocess
import sys

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Seed python, numpy and torch. Returns a seeded generator for DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_environment() -> dict:
    """Snapshot of everything needed to reproduce a run from scratch."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
    }
