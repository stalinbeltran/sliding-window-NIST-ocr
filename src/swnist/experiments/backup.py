"""Respaldo (git-ignored) de experimentos completos, incluyendo checkpoints."""

import shutil

from swnist import BACKUPS_DIR, EXPERIMENTS_DIR


def backup_experiment(exp_id: str) -> str:
    """Copia experiments/<id> (config, métricas, checkpoints) a backups/<id>."""
    src = EXPERIMENTS_DIR / exp_id
    dst = BACKUPS_DIR / exp_id
    BACKUPS_DIR.mkdir(exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return str(dst)
