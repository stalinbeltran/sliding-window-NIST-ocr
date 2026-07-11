"""Lanza la web app de entrenamiento.

Puerto: argumento 1 o variable de entorno SWNIST_PORT (por defecto 8000).
    python scripts/run_webapp.py [puerto]

Auto-recarga: el servidor se reinicia solo al cambiar el código (desactivar con
SWNIST_RELOAD=0). Los entrenamientos en curso NO sobreviven una recarga: no
edites código del paquete mientras haya un entrenamiento corriendo.
"""

import os
import sys

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SWNIST_PORT", "8000"))
    reload = os.environ.get("SWNIST_RELOAD", "1") != "0"
    uvicorn.run("swnist.webapp.main:app", host="127.0.0.1", port=port,
                reload=reload, reload_dirs=["src"])
