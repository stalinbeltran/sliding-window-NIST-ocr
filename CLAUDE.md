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
11. **Tests**: hay suite en `tests/` (`.\.venv\Scripts\python.exe -m pytest`, deps con
    `pip install -e .[dev]`). Correrla antes de cada commit que toque código. Los tests
    usan registro/backups en directorio temporal y datasets reducidos (no ensucian
    `experiments/` ni `backups/`).
12. **Sincronía dataset↔params**: cada dataset acepta solo sus propios parámetros
    (validado en `build_dataset` con mensaje claro). El frontend reemplaza
    `config.dataset.params` por los defaults del dataset al cambiar la selección
    (bug corregido el 2026-07-11: elegir `mnist_full` dejaba params de `mnist_windows`
    y reventaba con TypeError).
13. **Validación previa con razón**: toda restricción de compatibilidad
    (NN↔dataset, tamaños de ventana, re-entrenamiento) se valida ANTES de crear el
    experimento/evaluación (`src/swnist/validation.py`, usada por el manager) y la
    API responde 400 con un mensaje que explica el porqué y cómo corregirlo. Regla
    concreta: `model.window_size` del dimensionador debe coincidir con el tamaño de
    entrada efectivo del dataset (`mnist_full` → 28); el frontend lo sincroniza solo.
14. **Re-entrenamiento**: `config.init_from = <exp_id>` continúa desde los pesos
    `best.pt` de un experimento de la misma NN; la arquitectura (`config.model`) la
    dicta el experimento origen (el manager la reemplaza al validar). El optimizador
    parte de cero.
15. **Evaluaciones**: se lanzan desde la web app (pestaña Probar) sobre cualquier NN
    entrenada + dataset compatible + split (train/test) + límite opcional. Registro
    en `evaluations/<eval_id>/` (config.json, status.json con summary y matriz de
    confusión, results.jsonl por muestra con label/pred/conf/margin — el jsonl va
    git-ignored). `margin = p1 − p2`; "ambigua" = margin < umbral (0.2 por defecto).
    API: `POST/GET /api/evaluations`, `GET /api/evaluations/<id>/results` (filtros
    all|correct|failed|ambiguous, label, pred, paginado, miniaturas PNG base64).
16. **Datasets custom**: subconjuntos filtrados de una evaluación guardados en
    `custom_datasets/<nombre>.json` (versionado: base + índices reproducibles;
    nombre default automático y modificable). Semántica en `build_dataset`:
    `train=True` → el subconjunto; `train=False` → test completo del base. Heredan
    la compatibilidad del base y sirven para entrenar y evaluar. CRUD completo por
    API/UI; renombrar/eliminar se bloquea (409 con razón) si un experimento los
    referencia por nombre.
17. **CRUD de experimentos**: renombrar es un alias (`status.label`; el id nunca
    cambia porque otros experimentos lo referencian por id), duplicar copia el
    directorio completo con id nuevo, eliminar se bloquea si el experimento corre o
    si un secuenciador lo usa como dimensionador.
18. **Inspección de muestras**: `GET /api/datasets/<name>/samples` (miniaturas) y
    `POST /api/predict` (predicción de una muestra + `trace` paso a paso del
    deslizamiento: posiciones de la ventana y probs por paso; para el dimensionador
    se desliza su ventana sobre la imagen completa, para el secuenciador es su
    recorrido real). La UI (pestaña Probar) anima la ventana sobre el carácter y
    grafica la salida por paso. Modelos y datasets se cachean en
    `webapp/inference.py` (invalidar con `clear_caches()` al borrar/renombrar).

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
- `model.channels` es una **lista de longitud libre (≥1)**: un bloque conv
  (Conv3×3 + ReLU + MaxPool2) por elemento. El MaxPool se omite cuando la ventana ya
  no da más espacio; el pooling adaptativo final mantiene la salida en 3×3. Lista
  inválida → 400 con razón antes de crear el experimento (2026-07-11).
- Dataset por defecto del dimensionador en la web app: `mnist_full`
  (`model.window_size=28`), definido en `nn_registry.py` (2026-07-11).

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
  validation.py            ← compatibilidad NN↔dataset↔checkpoint (razones claras, 400)
  data/
    windows.py             ← utilidades de ventana deslizante (posiciones, extracción)
    datasets.py            ← MnistFull, MnistWindows, MnistSlidingSequences (+ display_item)
    custom.py              ← datasets custom: CustomDatasetStore (CRUD) + CustomSubset
    registry.py            ← catálogo de datasets (builtin+custom) + compatibilidad por NN
  models/
    dimensionador.py
    secuenciador.py
  training/
    common.py              ← split train/val, loaders reproducibles, checkpoints, init_from
    train_dimensionador.py
    train_secuenciador.py
  evaluation/
    registry.py            ← registro de evaluaciones + filtros de resultados
    runner.py              ← aplica una NN entrenada a un dataset, resultados por muestra
  experiments/
    registry.py            ← crear/listar/leer experimentos, métricas, CRUD (label/copy/delete)
    backup.py              ← respaldo a backups/ (git-ignored)
  webapp/
    main.py                ← FastAPI: API + estáticos
    manager.py             ← jobs (entrenamientos y evaluaciones) en hilos, con stop
    inference.py           ← miniaturas PNG, predict y trace (con caché de modelos/datasets)
    static/                ← index.html, app.js, style.css (sin CDNs; pestañas
                             Entrenar / Probar / Datasets)
tests/                     ← pytest: datasets (builtin y custom), modelos, entrenamiento
                             e2e, re-entrenamiento, evaluaciones y API completa
experiments/               ← registro versionado (config/métricas/estado por experimento)
                             * los checkpoints (experiments/*/checkpoints/) van git-ignored
evaluations/               ← registro de evaluaciones (results.jsonl git-ignored)
custom_datasets/           ← datasets custom versionados (base + índices)
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
**secuenciador** seleccionando ese experimento de dimensionador en el formulario. En la
pestaña **Probar**: evaluar cualquier NN entrenada sobre un dataset/split, explorar las
muestras (miniaturas; click → predicción con animación del deslizamiento y gráfica de la
salida por paso), filtrar aciertos/fallos/ambiguos y guardar el filtro como dataset
custom para re-entrenar (`init_from`) o volver a evaluar.

## Convenciones de entrenamiento

- Optimizador Adam; métricas: cross-entropy y accuracy (val y test); para el secuenciador
  además accuracy por paso (para ver cómo mejora la predicción a medida que la red "ve"
  más de la imagen) y `loss_mode`: `final` (solo último paso), `all` (todos los pasos,
  por defecto) o `ramp` (peso creciente con el paso).
- `val_fraction` se separa del train de MNIST con semilla fija; el test de MNIST solo se
  evalúa al final del entrenamiento (`final_test` en `status.json`).
- DataLoaders con `num_workers=0` (Windows) y generador seedeado.
