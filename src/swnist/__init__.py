"""swnist: sliding-window MNIST/NIST OCR (dimensionador CNN + secuenciador RNN)."""

__version__ = "0.1.0"

from pathlib import Path

# Repo root (src/swnist/__init__.py -> repo)
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
BACKUPS_DIR = ROOT / "backups"
DATA_DIR = ROOT / "data"
EVALUATIONS_DIR = ROOT / "evaluations"
CUSTOM_DATASETS_DIR = ROOT / "custom_datasets"
