# sliding-window-NIST-ocr

CNN configurable sobre MNIST: se entrena desde la web app y, una vez entrenada, se
**extraen sus features reconocidos** (las activaciones de cada capa convolucional) para
verlos como **matrices** y probarlos gráficamente sobre caracteres del propio dataset.

> **Historia**: el proyecto arrancó como un OCR por ventanas deslizantes con dos redes
> cooperantes (dimensionador + secuenciador). Ese código se borró por completo
> (commit `dad2a52 borrado todo`, 2026-07-13) y el proyecto se reinició con la
> arquitectura de una sola CNN descrita aquí. Las reglas de aquel diseño ya no aplican.

## Reglas permanentes del proyecto (NO repetir al usuario, aplicar siempre)

1. **Documentación para Claude**: toda decisión, convención o instrucción general queda
   registrada en este archivo (o en `docs/`) para no tener que repetirla. Si el usuario da
   una instrucción general nueva, se agrega aquí.
2. **Commits**: cada tarea que pida el usuario termina con un commit descriptivo.
3. **Backups**: se mantiene respaldo (git-ignored, en `backups/`) de cada experimento.
   El respaldo se hace automáticamente al terminar cada entrenamiento
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
   Misma semilla + misma config ⇒ mismos pesos (hay un test que lo comprueba).
9. **Métricas**: pérdida y accuracy de train/val por época y por lote, en
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
    `config.dataset.params` por los defaults del dataset al cambiar la selección.
13. **Validación previa con razón**: toda restricción se valida ANTES de crear el
    experimento (`src/swnist/validation.py`, usada por el manager) y la API responde 400
    con un mensaje que explica el porqué y cómo corregirlo. En particular: la arquitectura
    no puede agotar el mapa espacial (un conv o un pooling que no cabe en el mapa que le
    llega → 400 diciendo qué capa falla y con qué tamaño).
14. **Re-entrenamiento**: `config.init_from = <exp_id>` continúa desde los pesos `best.pt`
    de un experimento de la misma NN; **la arquitectura (`config.model`) la dicta el
    experimento origen** (el manager la reemplaza al validar), de modo que los pesos
    siempre cargan. El optimizador parte de cero.
15. **CRUD de experimentos**: renombrar es un alias (`status.label`; el id nunca cambia
    porque otros experimentos lo referencian con `init_from`), duplicar copia el
    directorio con id nuevo, eliminar se bloquea si el experimento corre o si otro lo usa
    como `init_from` (409 con razón).
16. **Lo más reciente primero**: el listado del registro va del más nuevo al más viejo (el
    id lleva timestamp → orden lexicográfico inverso), así los desplegables de la web app
    preseleccionan el último experimento entrenado.
17. **Consola Windows**: la salida por consola es cp1252. Cualquier CLI del proyecto debe
    forzar UTF-8 (`sys.stdout.reconfigure`) o una etiqueta con acentos rompe el listado.

## Arquitectura

### CNN de features (`features_cnn`)
- `src/swnist/models/features_cnn.py`
- **Todo configurable por JSON** (`config.model`):
  - `layers`: lista de longitud libre (1..8) — **N capas** convolucionales. Cada capa
    declara sus **kernels** (nº de filtros), `kernel_size`, `stride`, `padding`,
    `activation` (`relu|tanh|sigmoid|leaky_relu|none`) y su **pooling opcional**:
    `pool: null` (sin pooling) o `pool: {"type": "max"|"avg", "size": N, "stride": N|null}`.
  - `hidden_dim`: capa densa antes de la salida (0 = flatten conectado directo a la salida).
  - `dropout`, `num_classes` (10), `input_size` (28). La **capa de salida** es la densa
    final de `num_classes` logits.
- **API del modelo**:
  - `forward(x)` → logits (N, 10): clasificación del dígito.
  - `feature_maps(x)` → **lista con la salida de cada capa** (activación + pooling):
    (N, kernels, H, W). Cada canal es la respuesta de un kernel = un **feature
    reconocido**, y se envía a la web app como una **matriz H×W de números**.
  - `kernels()` → los pesos aprendidos de cada conv, también como matrices.
  - `layer_shapes()` → describe la arquitectura resultante (tamaño del mapa por capa).
- El tamaño espacial se calcula capa a capa (`spatial_trace`): si un conv o un pooling no
  cabe, se rechaza antes de entrenar (regla 13).

### Dataset
- `mnist` (`src/swnist/data/datasets.py`): imágenes 28×28 en [0, 1] + etiqueta 0-9.
  Único parámetro: `limit` (usar solo las primeras N muestras del split; útil para pruebas
  rápidas). La exploración de muestras y el predict usan siempre el split completo — el
  `limit` solo recorta el conjunto de entrenamiento.

### Visualización de los features
- `GET /api/datasets/<name>/samples` → miniaturas PNG (base64; encoder propio en
  `webapp/inference.py`, sin dependencias extra) para elegir un carácter.
- `POST /api/predict` `{experiment, dataset, split, index}` → predicción + `layers`: por
  cada capa, sus feature maps como matrices (con `min`/`max`/`mean` para normalizar el
  color). `max_maps` (default 64) trunca para no reventar el navegador.
- `GET /api/experiments/<id>/kernels` → los kernels aprendidos como matrices.
- Pestaña **Features** de la web app: eliges NN entrenada → carácter → se pintan los
  feature maps de cada capa como mapas de calor; click en cualquiera → su
  **representación matricial** completa (tabla de números).

## Estructura del repo

```
CLAUDE.md                  ← este archivo (reglas + arquitectura)
pyproject.toml             ← paquete swnist (src layout, pip install -e .)
scripts/run_webapp.py      ← lanza la web app (uvicorn, puerto 8000)
src/swnist/
  repro.py                 ← semillas y captura de entorno
  nn_registry.py           ← catálogo de NNs entrenables + config por defecto
  validation.py            ← validación previa de la config (razones claras, 400)
  data/
    datasets.py            ← MnistDigits
    registry.py            ← catálogo de datasets + compatibilidad por NN
  models/
    features_cnn.py        ← la CNN configurable + feature_maps() + kernels()
  training/
    common.py              ← split train/val, loaders reproducibles, checkpoints, init_from
    train_features_cnn.py  ← bucle de entrenamiento, métricas, backup
  experiments/
    registry.py            ← crear/listar/leer experimentos, métricas, CRUD
    backup.py              ← respaldo a backups/ (git-ignored)
  webapp/
    main.py                ← FastAPI: API + estáticos
    manager.py             ← jobs de entrenamiento en hilos, con stop
    inference.py           ← miniaturas PNG, predict con feature maps, kernels (con caché)
    static/                ← index.html, app.js, style.css (sin CDNs; pestañas
                             Entrenar / Experimentos / Features)
tests/                     ← pytest: modelo, validación, dataset, registro, entrenamiento
                             e2e, re-entrenamiento, extracción de features y API completa
experiments/               ← registro versionado (checkpoints/ va git-ignored)
backups/                   ← git-ignored, copia íntegra de cada experimento
data/                      ← git-ignored, descarga de MNIST (torchvision)
```

## Formato de un experimento (`experiments/<id>/`)

- `config.json` — config completa usada (nn, dataset+params, model, training, seed, init_from).
- `environment.json` — versiones, plataforma, commit git, disponibilidad CUDA.
- `status.json` — `status` (`created|running|completed|failed|stopped`), timestamps,
  `label`, `best` (mejor época por val_acc), `final_test`, `num_params`, `layer_shapes`,
  `error` si lo hubo.
- `metrics.jsonl` — una línea JSON por registro:
  - `{"type":"batch", "epoch":…, "step":…, "loss":…, "acc":…}` (cada `log_every` lotes)
  - `{"type":"epoch", "epoch":…, "train_loss":…, "train_acc":…, "val_loss":…,
     "val_acc":…, "seconds":…}`
- `checkpoints/best.pt`, `checkpoints/last.pt` — git-ignored.

Los ids tienen la forma `exp_YYYYMMDD_HHMMSS_<nn>`.

## Cómo correr

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]   # una vez
.\.venv\Scripts\python.exe scripts\run_webapp.py      # web app en http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m pytest                  # suite de tests
```

Flujo típico: **Entrenar** (elegir dataset, editar la arquitectura en el JSON, ver las
métricas en vivo) → **Experimentos** (renombrar/duplicar/eliminar/re-entrenar) →
**Features** (elegir la NN entrenada y un carácter del dataset, ver sus feature maps y la
matriz de cualquiera de ellos, o los kernels aprendidos).

## Convenciones de entrenamiento

- Optimizador Adam; pérdida cross-entropy; métricas: loss y accuracy de train/val por
  época y por lote.
- `val_fraction` se separa del train de MNIST con semilla fija; el test de MNIST solo se
  evalúa al final del entrenamiento (`final_test` en `status.json`).
- DataLoaders con `num_workers=0` (Windows) y generador seedeado.
- Cualquier fallo (incluido al construir dataset/modelo) deja el experimento en `failed`
  con la razón en `status.json`.
