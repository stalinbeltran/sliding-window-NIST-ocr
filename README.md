# sliding-window-NIST-ocr

OCR de dígitos por exploración secuencial con ventana deslizante:

- **Dimensionador**: CNN que extrae características de una región (ventana) de la imagen.
- **Secuenciador**: red recurrente que recibe la posición (x, y) y la salida del
  dimensionador, mantiene un estado interno retroalimentado y emite el dígito reconocido.

Todo el entrenamiento se realiza desde la web app:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_webapp.py
# → http://127.0.0.1:8000
```

Documentación completa (reglas, arquitectura, formato de experimentos): [CLAUDE.md](CLAUDE.md).
