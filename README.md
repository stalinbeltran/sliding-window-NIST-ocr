# sliding-window-NIST-ocr

OCR de dígitos por exploración secuencial con ventana deslizante:

- **Dimensionador**: CNN que extrae características de una región (ventana) de la imagen.
- **Secuenciador**: red recurrente que recibe la posición (x, y) y la salida del
  dimensionador, mantiene un estado interno retroalimentado y emite el dígito reconocido.

Todo el entrenamiento y las pruebas se realizan desde la web app:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_webapp.py
# → http://127.0.0.1:8000
```

La web app tiene tres pestañas:

- **Entrenar** — entrenar una NN desde cero o **re-entrenar** un experimento previo
  (`Continuar desde`), con métricas en vivo y CRUD de experimentos (renombrar,
  duplicar, eliminar).
- **Probar** — evaluar cualquier NN entrenada sobre cualquier dataset compatible
  (train/test), ver las muestras en miniatura, hacer click en una para aplicarla a la
  NN (con **animación del deslizamiento de la ventana** y gráfica de la salida por
  paso), filtrar aciertos/fallos/ambiguos y **guardar el filtro como dataset custom**.
- **Datasets** — catálogo de datasets base y CRUD de los datasets custom.

Toda combinación incompatible (p. ej. entrenar una NN de ventanas 14×14 con entradas
28×28) se rechaza con un mensaje que explica la razón.

Documentación completa (reglas, arquitectura, formato de experimentos): [CLAUDE.md](CLAUDE.md).
