"""Lanza la web app de entrenamiento.

Puerto: argumento 1 o variable de entorno SWNIST_PORT (por defecto 8000).
    python scripts/run_webapp.py [puerto]
"""

import os
import sys

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SWNIST_PORT", "8000"))
    uvicorn.run("swnist.webapp.main:app", host="127.0.0.1", port=port)
