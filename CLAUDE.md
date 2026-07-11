# sliding-window-NIST-ocr

Reconocimiento de dígitos MNIST/NIST mediante exploración secuencial de la imagen por
ventanas deslizantes, con dos redes neuronales cooperantes.

## Reglas permanentes del proyecto (NO repetir al usuario, aplicar siempre)

1. **Documentación para Claude**: toda decisión, convención o instrucción general queda
   registrada en este archivo (o en `docs/`) para no tener que repetirla. Si el usuario da
   una instrucción general nueva, se agrega aquí.
2. **Commits**: cada tarea que pida el usuario termina con un commit descriptivo.
3. **Backups**: se mantiene respaldo (git-ignored, en `backups/`) de cada experimento
   realizado. El respaldo se hace automáticamente al terminar cada entrenamiento
   (`swnist/experiments/backup.py`).
4. **Registro de experimentos**: cada experimento queda registrado en `experiments/<id>/`
   con configuración completa, métricas y estado. Es accesible mediante programación:
   - Python: `from swnist.experiments.registry import ExperimentRegistry`
   - CLI: `python -m swnist.experiments.registry` (lista todos los experimentos)
   - API web: `GET /api/experiments`
5. **Entrenamiento solo desde la web app**: todo entrenamiento se lanza desde la app web
   (`python scripts/run_webapp.py`, puerto 8000). No se crean scripts sueltos de
   entrenamiento manual.
6. **Selección de datasets en la web app**: la app permite seleccionar una NN y lista los
   datasets compatibles con esa NN (`swnist/data/registry.py` declara la compatibilidad).
7. **Stack**: Python + PyTorch (+ torchvision, FastAPI, uvicorn, numpy). Entorno en
   `.venv/` (git-ignored). En Windows el ejecutable es `.venv\Scripts\python.exe`.
8. **Reproducibilidad**: cada experimento guarda config completa (JSON), semilla, entorno
   (versiones de Python/torch, plataforma, commit git) para poder reproducirlo desde cero.
9. **Métricas**: se registran las métricas apropiadas (pérdida y accuracy de train/val por
   época y por lote; para el secuenciador además accuracy por paso de la secuencia) en
   `experiments/<id>/metrics.jsonl`, visualizables en vivo en la web app y legibles por
   Claude para depurar.
10. **Idioma**: el usuario se comunica en español; la UI y la documentación de alto nivel
    van en español. El código (identificadores, docstrings técnicos) puede ir en inglés.

## Arquitectura

Dos redes neuronales:

### 1. Dimensionador (CNN)
- `src/swnist/models/dimensionador.py`
- CNN que recibe una **región (ventana) de la entrada 2D** (p. ej. ventana 14×14 de una
  imagen MNIST 28×28) y produce:
  - `features(x)` → vector de características (`feature_dim`), que es lo que consume el
    secuenciador;
  - `forward(x)` → logits de clasificación (10 dígitos), usados para pre-entrenarlo.
- Usa pooling adaptativo, por lo que acepta ventanas de cualquier tamaño, pero **el
  tamaño de ventana con el que se entrenó queda en su config** y el secuenciador lo
  reutiliza obligatoriamente.

### 2. Secuenciador (NN recurrente / retroalimentada)
- `src/swnist/models/secuenciador.py`
- Entradas por paso: **posición (x, y)** actual (normalizada a [0,1]) + **salida del
  dimensionador** (features de la ventana en esa posición).
- Genera un **estado interno** que se retroalimenta a la misma red (celda GRU o Elman) y
  una **salida** con el dígito reconocido (logits por paso).
- Aprende a reconocer la entrada vista **por partes**, recorriendo la imagen con una
  ventana deslizante (orden raster por defecto).
- El dimensionador usado se referencia por **id de experimento** (checkpoint `best.pt`) y
  por defecto va congelado (`freeze_dimensionador: true`).

## Estructura del repo

```
CLAUDE.md                  ← este archivo (reglas + arquitectura)
pyproject.toml             ← paquete swnist (src layout, pip install -e .)
requirements.txt
scripts/run_webapp.py      ← lanza la web app (uvicorn, puerto 8000)
src/swnist/
  repro.py                 ← semillas y captura de entorno
  nn_registry.py           ← catálogo de NNs entrenables + configs por defecto
  data/
    windows.py             ← utilidades de ventana deslizante (posiciones, extracción)
    datasets.py            ← MnistFull, MnistWindows, MnistSlidingSequences
    registry.py            ← catálogo de datasets + compatibilidad por NN
  models/
    dimensionador.py
    secuenciador.py
  training/
    common.py              ← split train/val, loaders reproducibles, evaluación
    train_dimensionador.py
    train_secuenciador.py
  experiments/
    registry.py            ← crear/listar/leer experimentos, log de métricas
    backup.py              ← respaldo a backups/ (git-ignored)
  webapp/
    main.py                ← FastAPI: API + estáticos
    manager.py             ← jobs de entrenamiento en hilos, con stop
    static/                ← index.html, app.js, style.css (sin CDNs)
experiments/               ← registro versionado (config/métricas/estado por experimento)
                             * los checkpoints (experiments/*/checkpoints/) van git-ignored
backups/                   ← git-ignored, copia íntegra de cada experimento
data/                      ← git-ignored, descarga de MNIST (torchvision)
```

## Formato de un experimento (`experiments/<id>/`)

- `config.json` — config completa usada (nn, dataset+params, modelo, entrenamiento, seed).
- `environment.json` — versiones, plataforma, commit git, disponibilidad CUDA.
- `status.json` — `status` (`created|running|completed|failed|stopped`), timestamps,
  mejor resultado (`best`), resultado final en test (`final_test`), error si lo hubo.
- `metrics.jsonl` — una línea JSON por registro:
  - `{"type":"batch", "epoch":…, "step":…, "loss":…, "acc":…}` (cada `log_every` lotes)
  - `{"type":"epoch", "epoch":…, "train_loss":…, "train_acc":…, "val_loss":…,
     "val_acc":…, "seconds":…}` (+ `val_per_step_acc` en el secuenciador)
- `checkpoints/best.pt`, `checkpoints/last.pt` — git-ignored (estado del modelo,
  optimizador, config, época).

Los ids tienen la forma `exp_YYYYMMDD_HHMMSS_<nn>`.

## Cómo correr

```powershell
.\.venv\Scripts\python.exe -m pip install -e .   # una vez
.\.venv\Scripts\python.exe scripts\run_webapp.py # web app en http://127.0.0.1:8000
```

Flujo típico: entrenar primero un **dimensionador** desde la web app; luego entrenar un
**secuenciador** seleccionando ese experimento de dimensionador en el formulario.

## Convenciones de entrenamiento

- Optimizador Adam; métricas: cross-entropy y accuracy (val y test); para el secuenciador
  además accuracy por paso (para ver cómo mejora la predicción a medida que la red "ve"
  más de la imagen) y `loss_mode`: `final` (solo último paso), `all` (todos los pasos,
  por defecto) o `ramp` (peso creciente con el paso).
- `val_fraction` se separa del train de MNIST con semilla fija; el test de MNIST solo se
  evalúa al final del entrenamiento (`final_test` en `status.json`).
- DataLoaders con `num_workers=0` (Windows) y generador seedeado.
