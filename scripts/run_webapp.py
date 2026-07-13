"""Lanza la web app: python scripts/run_webapp.py [--port 8000]

El puerto también se puede fijar con la variable de entorno SWNIST_PORT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Web app de sliding-window-NIST-ocr")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SWNIST_PORT", "8000")),
        help="puerto (default 8000, o SWNIST_PORT)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Web app en http://{args.host}:{args.port}")
    # Sin reload: la auto-recarga sobre src/ mataría los entrenamientos en curso.
    uvicorn.run("swnist.webapp.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
