"""Respaldo (git-ignored) de experimentos completos, incluyendo checkpoints."""

import shutil
from pathlib import Path

from swnist import BACKUPS_DIR, EXPERIMENTS_DIR


def backup_experiment(
    exp_id: str,
    experiments_root: Path | None = None,
    backups_root: Path | None = None,
) -> str:
    """Copia experiments/<id> (config, métricas, checkpoints) a backups/<id>.

    Los roots son parametrizables para que los tests usen directorios temporales.
    """
    src = (experiments_root or EXPERIMENTS_DIR) / exp_id
    dst_root = backups_root or BACKUPS_DIR
    dst = dst_root / exp_id
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return str(dst)
