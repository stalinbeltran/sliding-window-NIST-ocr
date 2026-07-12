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
    entrada efectivo del dataset (`params.window_size` merged sobre los defaults);
    el frontend lo sincroniza solo — también al editar `dataset.params.window_size`
    a mano en el JSON y al pulsar Entrenar (`syncDimensionadorWindow`; bug del
    2026-07-11: solo se sincronizaba al cambiar el desplegable de dataset, y editar
    los params a mano dejaba `model.window_size` desfasado → 400). Con `init_from`
    no se toca (la arquitectura la dicta el experimento origen). También se validan los valores:
    `window_size` ∈ [1, 28], `windows_per_image`/`stride` ≥ 1. El dimensionador
    de un secuenciador debe tener `window_size` < 28: con ventana 28 el recorrido
    colapsa a 1 solo paso y no hay secuenciación (400 con razón). La config del
    secuenciador guarda en `dataset.params.window_size` el valor efectivo (el del
    dimensionador), no el del formulario (bug detectado el 2026-07-11: un
    secuenciador con dimensionador de ventana 28 "secuenciaba" en 1 paso aunque
    la config dijera `window_size=5`).
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
19. **Ventana adaptable en cualquier dataset** (2026-07-11): todo dataset sirve de
    base para entrenar con cualquier tamaño de ventana. `mnist_full` acepta
    `window_size` (default 28 = imagen completa) y `windows_per_image` (default 1);
    con `window_size < 28` produce ventanas aleatorias (misma lógica que
    `mnist_windows`, que ahora es subclase de `MnistFull`), pero la muestra visible
    sigue siendo la imagen completa (la UI anima la ventana encima). Los datasets
    custom exponen como `defaults` los params efectivos de su base y admiten
    sobrescribir `window_size` (re-entrenar las mismas muestras con otra ventana);
    cambiar `windows_per_image` en un custom se rechaza (400) porque invalidaría
    los índices guardados.

20. **Stride del secuenciador fijado de extremo a extremo** (2026-07-11): la trayectoria
    (posiciones (x, y) por paso) es ENTRADA de la red, así que el stride efectivo queda
    registrado en la config al entrenar (`validate_train_config`) y las evaluaciones y
    el predict la reutilizan obligatoriamente
    (`swnist.validation.sequence_effective_params`; evaluar con otro stride → 400 con
    razón; runner y predict lo fijan también por defensa, y la config de la evaluación
    guarda los valores efectivos). Bug original: secuenciador entrenado con stride 5
    (36 pasos) evaluado con el stride 7 de los defaults del formulario (25 pasos) caía
    de acc 0.91 a 0.56. `GET /api/datasets/<name>/slide` devuelve el recorrido
    (posiciones, pasos, notas de solape/huecos) y la pestaña Datasets tiene un
    visualizador que anima ventana+stride (editables, solo visualización) sobre el mapa
    de píxeles de una muestra; el trace de predict incluye el stride efectivo. Los
    datasets de ventanas (mnist_full/mnist_windows) solo usan su stride con
    sampling='raster' (regla 24); con sampling='random' (default) es inerte.

21. **Trayectoria guiada por el contenido** (2026-07-11, branch
    `feature/trayectoria-por-contenido`): `mnist_contour_sequences` hace que la ventana
    siga la 'forma' del carácter en vez de escanear el área en raster: tinta binarizada
    → esqueleto de ~1 px (Zhang-Suen, numpy puro) → camino ordenado (vecino más cercano
    desde un extremo del trazo) → remuestreo a `num_steps` posiciones fijas
    (`src/swnist/data/trajectory.py`). T = `num_steps` es constante (batching y accuracy
    por paso lo requieren); la trayectoria depende de cada muestra pero es determinista
    (sin RNG; cacheada por índice en el dataset). `num_steps` es un parámetro de
    trayectoria como el stride (regla 20): se fija en la config al entrenar y
    evaluaciones/predict lo reutilizan obligatoriamente; además, evaluar un secuenciador
    con un dataset de OTRO tipo de recorrido (raster↔trazo) es 400 con razón
    (`TRAJECTORY_KEYS` en `validation.py`). Los datasets de secuencias exponen
    `trajectory(idx)` (posiciones (top, left) de la muestra; `CustomSubset` delega) y lo
    usan el trace de predict y `GET /api/datasets/<name>/slide` (que acepta
    `split`/`index`/`num_steps` para los datasets de trazo, porque el recorrido cambia
    con la muestra; el visualizador de la pestaña Datasets lo anima igual).

22. **Clase "no hay nada en este recuadro"** (2026-07-12): los datasets de ventanas
    (`mnist_full`/`mnist_windows` y sus customs) aceptan `empty_fraction` ∈ [0, 1):
    esa fracción de las muestras es una ventana SIN píxeles activos (gris original ≤
    `EMPTY_INK_THRESHOLD` = 0.05) etiquetada `EMPTY_LABEL` = 10, tomada de una región
    vacía de la misma imagen (o fondo puro si no la hay); determinista dado (seed, idx)
    y con `empty_fraction=0` las muestras antiguas se reproducen idénticas (no se
    consume RNG extra). Con `empty_fraction > 0` el dimensionador necesita
    `model.num_classes=11`: `validate_train_config` lo fija en la config (y el frontend
    lo sincroniza en el JSON, `syncDimensionadorWindow`); `init_from` de un modelo de
    10 clases → 400 con razón, igual que evaluar un modelo de 10 clases con un dataset
    con vacíos. La pestaña Probar propaga el `empty_fraction` del entrenamiento a las
    evaluaciones; la matriz de confusión se dimensiona con `num_classes` del modelo;
    la UI muestra la clase 10 como "∅" (también en el filtro por etiqueta). Cambiar
    `empty_fraction` en un dataset custom se rechaza (re-etiquetaría sus muestras).
    El secuenciador no cambia: sigue clasificando 0–9, pero las features del
    dimensionador ahora le comunican que la región observada está vacía.
    **NOTA (2026-07-12):** regla retirada. El dimensionador dejó de clasificar dígitos
    y de usar datasets NIST (regla 25): ya no existe la clase "no hay nada" ni
    `empty_fraction` (mnist_full/mnist_windows eliminados).

23. **Uniformización del grosor de trazo** (2026-07-12): todos los datasets (builtin y
    custom) aceptan `stroke_width` ∈ [0, 28] (`src/swnist/data/strokes.py`): 0 (default)
    deja el trazo original; N > 0 redibuja el carácter con trazo de ~N px de grosor
    uniforme (tinta → esqueleto Zhang-Suen reutilizado de `trajectory.py` →
    re-entintado por distancia al esqueleto con borde anti-alias de 1 px), de modo que
    un trazo grueso y uno delgado del mismo carácter quedan iguales ("dataset más
    uniforme"). Determinista (sin RNG; caché uint8 por índice en `StrokeUniformizer`);
    ventanas, ventanas vacías (`empty_fraction`), trayectoria por el trazo y
    `display_item` salen todos de la MISMA imagen transformada; una imagen sin tinta
    vuelve intacta. Con `stroke_width=0` las muestras antiguas se reproducen idénticas.
    En un custom es sobrescribible (no invalida índices ni etiquetas). UI: el param
    aparece en `dataset.params` en Entrenar como cualquier otro; la pestaña Probar
    propaga el `stroke_width` del entrenamiento a las evaluaciones (dimensionador y
    secuenciador; el servidor NO bloquea un valor distinto explícito — evaluar con otro
    grosor es un experimento de robustez legítimo); el visualizador de la pestaña
    Datasets tiene un input `stroke_width` que previsualiza la transformación sobre la
    muestra elegida, y `GET /api/datasets/<name>/slide` lo acepta (el recorrido por el
    trazo se calcula sobre la imagen transformada; fuera de rango → 400 con razón).

24. **Muestreo raster de ventanas para el dimensionador** (2026-07-12): los datasets de
    ventanas (`mnist_full`/`mnist_windows` y sus customs) aceptan `sampling` ∈
    {`random`, `raster`} (default `random`, el comportamiento de siempre). Con
    `sampling='raster'` cada imagen produce la grilla completa de posiciones
    `window_size`+`stride` en orden raster (`grid_positions`, la misma grilla que
    recorre el secuenciador y el trace de predict): `windows_per_image` se ignora (el
    número de ventanas por imagen lo dicta la grilla) y no se consume RNG (determinista
    dado (params, idx)); `empty_fraction` funciona igual en ambos modos. Con
    `sampling='random'` el `stride` es inerte. Validación: `sampling` con otro valor →
    400 con razón; en un custom no se puede cambiar `sampling` (los índices apuntarían
    a otras ventanas) y, si el base es raster, tampoco `window_size`/`stride` (definen
    la grilla y remapearían los índices; con base random `window_size` sí se puede,
    regla 19). La pestaña Probar propaga `sampling` (y su `stride` si es raster) del
    entrenamiento a las evaluaciones; en `GET /api/datasets/<name>/slide` el
    visualizador muestra la grilla y aclara cuándo el stride del dataset aplica
    (`dataset_uses_stride` = raster) o es inerte (random).
    **NOTA (2026-07-12):** retirada para el dimensionador junto con los datasets NIST
    (regla 25). El muestreo raster ya no existe (mnist_full/mnist_windows eliminados);
    los datasets de secuencias conservan su `stride` como parámetro de trayectoria.

25. **Dimensionador de descriptores geométricos sobre trazos sintéticos** (2026-07-12,
    branch `feature/dimensionador-descriptores`): el dimensionador dejó de clasificar
    dígitos y ahora **describe el trazo** de su ventana. Deja de usar datasets NIST:
    su único dataset es `synthetic_strokes` (`src/swnist/data/synthstrokes.py`), que
    genera rectas/curvas (continuas/entrecortadas) sobre una ventana cuadrada (default
    5×5) con un vector de **descriptores geométricos como target**, conocido por
    construcción. El contrato de los 7 descriptores vive en `src/swnist/descriptors.py`
    (única fuente de verdad para dataset, modelo, entrenamiento, evaluación y UI):
    `straightness` (1=recta), `horizontal` (|cos θ|), `vertical` (|sin θ|), `angle_sin2`
    y `angle_cos2` (el ángulo como par sin/cos de 2θ, sin salto en 0°/180°),
    `continuity` (1=continua) e `ink` (**cantidad de tinta** = cobertura/gris medio de
    la ventana; 0 = vacía). `forward` regresa los crudos (pérdida MSE por canal),
    `features` = `activate(forward)` los entrega en rango natural al secuenciador
    (`feature_dim` fijo = 7). El **conjunto de muestras se define por rangos
    JSON-editables** desde la UI (num_samples, curved_fraction, broken_fraction, rangos
    de ángulo/longitud/curvatura/guiones, stroke_width y empty_fraction). La
    **generación de trazos nunca produce una ventana vacía** (siempre hay tinta; sí
    trazos cortos); las únicas ventanas vacías son las de `empty_fraction` (default 0.1):
    100% en blanco con `ink=0` y la **geometría enmascarada** de la pérdida (los 6
    canales de `GEOMETRY` no se supervisan ahí, solo `ink`), para que el secuenciador
    —que sí encuentra fondo al deslizar sobre un dígito— reciba la señal "aquí no hay
    nada". El `mask` por canal es el 3er elemento de cada muestra
    (`window, target, mask`). Determinista dado (seed, train, idx); con
    `empty_fraction=0` nada se enmascara. La métrica registrada como `val_acc` es un
    score combinado ∈ [0,1] (media de 1−error **sobre los canales supervisados** de cada
    muestra); en `metrics.jsonl` van además `angle_err_deg`, `straightness_mae`,
    `continuity_mae`, `ink_mae`. La evaluación resume por **categoría discreta** (recta/
    curva × continua/entrecortada + **`vacío`** cuando `ink < EMPTY_INK`, matriz 5×5)
    para reutilizar filtros/grilla de Probar, y añade `angle_mae_deg` y `descriptor_mae`
    (sobre canales supervisados) al summary; el predict del dimensionador devuelve los
    descriptores (los canales enmascarados del target salen como `null`), no una
    distribución de dígitos, y no tiene trace (la muestra ES la ventana). El
    **secuenciador no cambia** salvo que su feature_dim pasa a 7: sigue clasificando 0–9
    consumiendo las features del dimensionador (ahora interpretables, incluida la tinta)
    al recorrer dígitos MNIST reales. Consecuencia: reglas 22 (clase vacío) y 24
    (sampling raster) quedan retiradas para el dimensionador, y la parte de la regla 19
    sobre ventanas de mnist_full/mnist_windows ya no aplica (esos datasets se eliminaron).

## Arquitectura

Dos redes neuronales:

### 1. Dimensionador (CNN)
- `src/swnist/models/dimensionador.py`
- CNN que recibe una **región (ventana) de la entrada 2D** (ventana cuadrada, default
  5×5) y la **describe** con un vector de descriptores geométricos interpretables
  (ver regla 25 y `src/swnist/descriptors.py`): rectitud (recta/curva), horizontal,
  vertical, ángulo del trazo (par sin 2θ/cos 2θ), continuidad (continua/entrecortada)
  y cantidad de tinta (`ink`; 0 = ventana vacía). Ya **no clasifica dígitos**.
  - `forward(x)` → descriptores **crudos** (6), usados por la pérdida (MSE por canal).
  - `features(x)` = `activate(forward(x))` → los mismos descriptores en su **rango
    natural**; ese es exactamente el vector que consume el secuenciador (pipeline
    interpretable). `feature_dim` queda fijo = nº de descriptores (6).
- Usa pooling adaptativo, por lo que acepta ventanas de cualquier tamaño, pero **el
  tamaño de ventana con el que se entrenó queda en su config** y el secuenciador lo
  reutiliza obligatoriamente.
- `model.channels` es una **lista de longitud libre (≥1)**: un bloque conv
  (Conv3×3 + ReLU + MaxPool2) por elemento. El MaxPool se omite cuando la ventana ya
  no da más espacio; el pooling adaptativo final mantiene la salida en 3×3. Lista
  inválida → 400 con razón antes de crear el experimento. `model.hidden_dim` (default
  32) es el ancho de la capa densa antes de la cabeza de descriptores.
- **Único dataset del dimensionador**: `synthetic_strokes` (rectas/curvas sintéticas,
  regla 25), `model.window_size=5` por defecto (`nn_registry.py`). Ya **no se usan
  datasets NIST** (mnist_full/mnist_windows) para el dimensionador (2026-07-12).

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
  descriptors.py           ← contrato de los 7 descriptores del dimensionador (regla 25)
  nn_registry.py           ← catálogo de NNs entrenables + configs por defecto
  validation.py            ← compatibilidad NN↔dataset↔checkpoint (razones claras, 400)
  data/
    windows.py             ← utilidades de ventana deslizante (posiciones, extracción)
    trajectory.py          ← trayectoria por el trazo: esqueleto → camino → num_steps (regla 21)
    strokes.py             ← uniformización del grosor de trazo (stroke_width, regla 23)
    synthstrokes.py        ← dataset sintético de rectas/curvas del dimensionador (regla 25)
    datasets.py            ← MnistSlidingSequences, MnistContourSequences (secuenciador)
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
