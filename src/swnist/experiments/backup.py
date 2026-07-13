"""Respaldo de experimentos en `backups/` (git-ignored), automático al terminar
cada entrenamiento (regla 3)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BACKUPS_DIR = ROOT / "backups"


def backup_experiment(
    exp_dir: Path | str,
    *,
    backups_dir: Path | str | None = None,
) -> Path:
    """Copia íntegra del directorio del experimento. Devuelve la ruta del respaldo."""
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"no existe el experimento a respaldar: {exp_dir}")
    base = Path(backups_dir) if backups_dir else DEFAULT_BACKUPS_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / f"{exp_dir.name}__{stamp}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(exp_dir, target)
    return target
