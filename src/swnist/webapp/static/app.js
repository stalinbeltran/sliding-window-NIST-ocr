"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  nns: [],
  datasets: [],          // builtin + custom, tal como los lista la API
  experiments: [],
  evaluations: [],
  training: null,        // { id, since, epochs: [], batches: [] }
  currentEval: null,     // evaluación abierta en la pestaña Probar
  evalFilter: { outcome: "all", label: "", pred: "" },
  evalOffset: 0,
  prediction: null,
};

const PAGE = 48;

// --- utilidades -------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

const fmt = (value, digits = 4) =>
  typeof value === "number" ? value.toFixed(digits) : "—";

const readConfig = () => JSON.parse($("config-json").value);
const writeConfig = (config) => ($("config-json").value = JSON.stringify(config, null, 2));

function describeArchitecture(config) {
  return (config?.model?.layers || [])
    .map((layer) => {
      const pool = layer.pool ? `+${layer.pool.type}${layer.pool.size}` : "";
      return `${layer.kernels}@${layer.kernel_size}x${layer.kernel_size}${pool}`;
    })
    .join(" → ");
}

function datasetByName(name) {
  return state.datasets.find((ds) => ds.name === name);
}

function fillSelect(select, options, { keep = true } = {}) {
  const previous = select.value;
  select.innerHTML = options
    .map((o) => `<option value="${o.value}">${o.text}</option>`)
    .join("");
  if (keep && options.some((o) => o.value === previous)) select.value = previous;
}

// --- pestañas ---------------------------------------------------------------

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    button.classList.add("active");
    $(`tab-${button.dataset.tab}`).classList.add("active");
    if (button.dataset.tab === "experimentos") loadExperiments();
    if (button.dataset.tab === "probar") openProbar();
    if (button.dataset.tab === "datasets") loadDatasets();
    if (button.dataset.tab === "features") refreshFeatureSelectors();
  });
});

const goToTab = (name) => document.querySelector(`.tab[data-tab="${name}"]`).click();

// --- catálogos --------------------------------------------------------------

async function loadCatalogs() {
  state.nns = await api("/api/nns");
  fillSelect(
    $("nn-select"),
    state.nns.map((nn) => ({ value: nn.name, text: nn.label })),
    { keep: false }
  );
  await onNnChange();
  await loadExperiments();
}

async function refreshDatasets(nn) {
  state.datasets = await api(`/api/datasets${nn ? `?nn=${nn}` : ""}`);
  return state.datasets;
}

async function onNnChange() {
  const nn = state.nns.find((item) => item.name === $("nn-select").value);
  $("nn-desc").textContent = nn.description;
  await refreshDatasets(nn.name);
  fillSelect(
    $("dataset-select"),
    state.datasets.map((ds) => ({ value: ds.name, text: ds.label })),
    { keep: false }
  );
  writeConfig(structuredClone(nn.default_config));
  $("dataset-select").value = nn.default_config.dataset.name;
}

// Al cambiar de dataset se reemplazan los params por SUS defaults (regla 12).
function onDatasetChange() {
  const config = readConfig();
  const dataset = datasetByName($("dataset-select").value);
  config.dataset = { name: dataset.name, params: structuredClone(dataset.defaults) };
  writeConfig(config);
}

// --- editor de arquitectura -------------------------------------------------

function addLayer() {
  const config = readConfig();
  const last = config.model.layers[config.model.layers.length - 1];
  config.model.layers.push(
    structuredClone(last) || {
      kernels: 8, kernel_size: 3, stride: 1, padding: 1, activation: "relu", pool: null,
    }
  );
  writeConfig(config);
}

function removeLayer() {
  const config = readConfig();
  if (config.model.layers.length <= 1) {
    $("train-error").textContent = "La CNN necesita al menos 1 capa convolucional.";
    return;
  }
  config.model.layers.pop();
  writeConfig(config);
}

// --- entrenamiento ----------------------------------------------------------

async function startTraining() {
  $("train-error").textContent = "";
  let config;
  try {
    config = readConfig();
  } catch (error) {
    $("train-error").textContent = `JSON inválido: ${error.message}`;
    return;
  }
  config.init_from = $("initfrom-select").value || null;

  try {
    const experiment = await api("/api/experiments", {
      method: "POST",
      body: JSON.stringify({ config, label: $("label-input").value || null }),
    });
    state.training = { id: experiment.id, since: 0, epochs: [], batches: [] };
    $("btn-stop").disabled = false;
    $("epochs-table").querySelector("tbody").innerHTML = "";
    pollMetrics();
  } catch (error) {
    $("train-error").textContent = error.message;
  }
}

async function stopTraining() {
  if (!state.training) return;
  try {
    await api(`/api/experiments/${state.training.id}/stop`, { method: "POST" });
  } catch (error) {
    $("train-error").textContent = error.message;
  }
}

async function pollMetrics() {
  if (!state.training) return;
  const { id, since } = state.training;
  let payload;
  try {
    payload = await api(`/api/experiments/${id}/metrics?since=${since}`);
  } catch (error) {
    $("train-error").textContent = error.message;
    return;
  }
  state.training.since = payload.next;

  for (const record of payload.records) {
    if (record.type === "epoch") {
      state.training.epochs.push(record);
      addEpochRow(record);
    } else {
      state.training.batches.push(record);
    }
  }
  drawChart();

  $("live-header").innerHTML = payload.running
    ? `<b>${id}</b> · corriendo · ${state.training.epochs.length} época(s)`
    : `<b>${id}</b> · ${payload.status}`;

  if (payload.running) {
    setTimeout(pollMetrics, 1000);
  } else {
    $("btn-stop").disabled = true;
    const experiment = await api(`/api/experiments/${id}`);
    if (experiment.final_test) {
      $("live-header").innerHTML += ` · test acc <b>${fmt(experiment.final_test.acc)}</b>`;
    }
    if (experiment.error) $("train-error").textContent = experiment.error;
    state.training = null;
    loadExperiments();
  }
}

function addEpochRow(record) {
  const row = document.createElement("tr");
  row.innerHTML = `<td>${record.epoch}</td><td>${fmt(record.train_loss)}</td>
    <td>${fmt(record.train_acc)}</td><td>${fmt(record.val_loss)}</td>
    <td>${fmt(record.val_acc)}</td><td>${fmt(record.seconds, 1)}</td>`;
  $("epochs-table").querySelector("tbody").appendChild(row);
}

function drawChart() {
  const canvas = $("live-chart");
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (!state.training) return;

  const { batches, epochs } = state.training;
  const pad = 30;
  ctx.strokeStyle = "#2c3444";
  ctx.strokeRect(pad, 10, width - pad - 10, height - pad - 10);
  ctx.fillStyle = "#8b95a7";
  ctx.font = "10px Consolas";
  ctx.fillText("1.0", 4, 18);
  ctx.fillText("0.0", 4, height - pad + 2);

  const plot = (points, color) => {
    if (points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((value, i) => {
      const x = pad + (i / (points.length - 1)) * (width - pad - 10);
      const y = 10 + (1 - Math.min(Math.max(value, 0), 1)) * (height - pad - 20);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  plot(batches.map((b) => Math.min(b.loss / 2.5, 1)), "#ff9f43");
  plot(batches.map((b) => b.acc), "#4da3ff");
  plot(epochs.map((e) => e.val_acc), "#3ddc84");

  ctx.fillStyle = "#ff9f43"; ctx.fillText("loss (lote)", pad + 6, 22);
  ctx.fillStyle = "#4da3ff"; ctx.fillText("acc (lote)", pad + 80, 22);
  ctx.fillStyle = "#3ddc84"; ctx.fillText("val acc (época)", pad + 150, 22);
}

// --- experimentos -----------------------------------------------------------

async function loadExperiments() {
  state.experiments = await api("/api/experiments");
  renderExperiments();
  renderInitFrom();
}

const trainedExperiments = () => state.experiments.filter((exp) => exp.has_checkpoint);

function renderInitFrom() {
  $("initfrom-select").innerHTML =
    `<option value="">(desde cero)</option>` +
    trainedExperiments()
      .map((exp) => `<option value="${exp.id}">${exp.label} · ${exp.id}</option>`)
      .join("");
}

function renderExperiments() {
  const body = $("experiments-table").querySelector("tbody");
  body.innerHTML = "";
  for (const exp of state.experiments) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${exp.label}</td>
      <td><code>${exp.id}</code></td>
      <td>${exp.running ? "corriendo" : exp.status}</td>
      <td>${describeArchitecture(exp.config)}</td>
      <td>${exp.config.dataset.name}</td>
      <td>${fmt(exp.best?.val_acc)}</td>
      <td>${fmt(exp.final_test?.acc)}</td>
      <td class="actions"></td>`;
    const actions = row.querySelector(".actions");

    const probar = document.createElement("button");
    probar.className = "mini primary";
    probar.textContent = "probar";
    probar.disabled = !exp.has_checkpoint;
    probar.title = exp.has_checkpoint
      ? "Evaluar esta NN sobre un dataset y filtrar los resultados"
      : "Todavía no tiene checkpoint: entrénala primero";
    probar.onclick = () => {
      goToTab("probar");
      $("eval-exp").value = exp.id;
      onEvalExperimentChange();
    };

    const features = document.createElement("button");
    features.className = "mini";
    features.textContent = "features";
    features.disabled = !exp.has_checkpoint;
    features.onclick = () => {
      goToTab("features");
      $("feat-exp").value = exp.id;
    };

    const rename = document.createElement("button");
    rename.className = "mini";
    rename.textContent = "renombrar";
    rename.onclick = async () => {
      const label = prompt("Nuevo nombre", exp.label);
      if (label) {
        await api(`/api/experiments/${exp.id}`, {
          method: "PATCH",
          body: JSON.stringify({ label }),
        });
        loadExperiments();
      }
    };

    const duplicate = document.createElement("button");
    duplicate.className = "mini";
    duplicate.textContent = "duplicar";
    duplicate.onclick = async () => {
      await api(`/api/experiments/${exp.id}/duplicate`, { method: "POST" });
      loadExperiments();
    };

    const retrain = document.createElement("button");
    retrain.className = "mini";
    retrain.textContent = "re-entrenar";
    retrain.disabled = !exp.has_checkpoint;
    retrain.onclick = () => {
      goToTab("entrenar");
      writeConfig({ ...exp.config, init_from: exp.id });
      $("initfrom-select").value = exp.id;
      $("label-input").value = `${exp.label} (re-entrenado)`;
    };

    const remove = document.createElement("button");
    remove.className = "mini";
    remove.textContent = "eliminar";
    remove.onclick = async () => {
      if (!confirm(`¿Eliminar ${exp.id}?`)) return;
      try {
        await api(`/api/experiments/${exp.id}`, { method: "DELETE" });
        loadExperiments();
      } catch (error) {
        alert(error.message);
      }
    };

    actions.append(probar, features, rename, duplicate, retrain, remove);
    body.appendChild(row);
  }
}

// --- probar (evaluaciones) --------------------------------------------------

async function openProbar() {
  await loadExperiments();
  await refreshDatasets($("nn-select").value);
  fillSelect(
    $("eval-exp"),
    trainedExperiments().map((exp) => ({
      value: exp.id,
      text: `${exp.label} · ${exp.id}`,
    }))
  );
  onEvalExperimentChange();
  await loadEvaluations();
}

function onEvalExperimentChange() {
  const exp = state.experiments.find((e) => e.id === $("eval-exp").value);
  fillSelect(
    $("eval-dataset"),
    state.datasets.map((ds) => ({ value: ds.name, text: ds.label }))
  );
  if (exp && state.datasets.some((ds) => ds.name === exp.config.dataset.name)) {
    $("eval-dataset").value = exp.config.dataset.name;
  }
  onEvalDatasetChange();
}

function onEvalDatasetChange() {
  const dataset = datasetByName($("eval-dataset").value);
  const split = $("eval-split");
  if (dataset?.is_custom) {
    split.innerHTML = `<option value="train">el subconjunto guardado (${dataset.size})</option>
      <option value="test">test completo del base (${dataset.base})</option>`;
    $("eval-split-hint").textContent =
      `«${dataset.name}» es un subconjunto de ${dataset.base}/${dataset.split}: evalúa esas ` +
      `${dataset.size} muestras, o el test completo del base.`;
  } else {
    split.innerHTML = `<option value="test">test</option><option value="train">train</option>`;
    $("eval-split-hint").textContent =
      "Evalúa la NN sobre el dataset elegido; luego podrás filtrar por clase y por " +
      "resultado, y guardar el filtro como dataset nuevo.";
  }
}

async function evaluate() {
  $("eval-error").textContent = "";
  const limit = $("eval-limit").value;
  try {
    const evaluation = await api("/api/evaluations", {
      method: "POST",
      body: JSON.stringify({
        config: {
          experiment: $("eval-exp").value,
          dataset: $("eval-dataset").value,
          split: $("eval-split").value,
          limit: limit ? parseInt(limit, 10) : null,
          ambiguous_threshold: parseFloat($("eval-threshold").value),
        },
      }),
    });
    await waitForEvaluation(evaluation.id);
  } catch (error) {
    $("eval-error").textContent = error.message;
  }
}

async function waitForEvaluation(evalId) {
  const evaluation = await api(`/api/evaluations/${evalId}`);
  await loadEvaluations();
  if (evaluation.running || evaluation.status === "created") {
    $("eval-detail").innerHTML = `<p class="muted">Evaluando ${evalId}…</p>`;
    setTimeout(() => waitForEvaluation(evalId), 700);
  } else if (evaluation.status === "failed") {
    $("eval-error").textContent = evaluation.error;
  } else {
    openEvaluation(evalId);
  }
}

async function loadEvaluations() {
  state.evaluations = await api("/api/evaluations");
  const body = $("evaluations-table").querySelector("tbody");
  body.innerHTML = "";
  for (const evaluation of state.evaluations) {
    const summary = evaluation.summary || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${evaluation.label}</td>
      <td><code>${evaluation.experiment}</code></td>
      <td>${evaluation.dataset} / ${evaluation.split}</td>
      <td>${evaluation.running ? "corriendo" : evaluation.status}</td>
      <td>${fmt(summary.accuracy)}</td>
      <td>${summary.failed ?? "—"}</td>
      <td>${summary.ambiguous ?? "—"}</td>
      <td class="actions"></td>`;
    const actions = row.querySelector(".actions");

    const open = document.createElement("button");
    open.className = "mini primary";
    open.textContent = "ver resultados";
    open.disabled = evaluation.status !== "completed";
    open.onclick = () => openEvaluation(evaluation.id);

    const rename = document.createElement("button");
    rename.className = "mini";
    rename.textContent = "renombrar";
    rename.onclick = async () => {
      const label = prompt("Nuevo nombre", evaluation.label);
      if (label) {
        await api(`/api/evaluations/${evaluation.id}`, {
          method: "PATCH",
          body: JSON.stringify({ label }),
        });
        loadEvaluations();
      }
    };

    const remove = document.createElement("button");
    remove.className = "mini";
    remove.textContent = "eliminar";
    remove.onclick = async () => {
      if (!confirm(`¿Eliminar ${evaluation.id}?`)) return;
      try {
        await api(`/api/evaluations/${evaluation.id}`, { method: "DELETE" });
        if (state.currentEval?.id === evaluation.id) {
          state.currentEval = null;
          $("eval-detail").innerHTML = "";
        }
        loadEvaluations();
      } catch (error) {
        alert(error.message);
      }
    };

    actions.append(open, rename, remove);
    body.appendChild(row);
  }
}

async function openEvaluation(evalId) {
  state.currentEval = await api(`/api/evaluations/${evalId}`);
  state.evalFilter = { outcome: "all", label: "", pred: "" };
  state.evalOffset = 0;
  renderEvaluation();
  await loadResults();
}

function renderEvaluation() {
  const evaluation = state.currentEval;
  const summary = evaluation.summary;
  const classes = evaluation.num_classes || 10;

  const head = Array.from({ length: classes }, (_, j) => `<th>${j}</th>`).join("");
  const rows = evaluation.confusion
    .map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0);
      const cells = row
        .map((count, j) => {
          const intensity = total ? count / total : 0;
          const bg = i === j
            ? `rgba(61, 220, 132, ${0.15 + 0.75 * intensity})`
            : `rgba(255, 107, 107, ${count ? 0.15 + 0.75 * intensity : 0})`;
          return `<td class="cell" style="background:${bg}" data-label="${i}" data-pred="${j}"
            title="reales ${i} predichos como ${j}">${count || ""}</td>`;
        })
        .join("");
      return `<tr><th>${i}</th>${cells}</tr>`;
    })
    .join("");

  $("eval-detail").innerHTML = `
    <div class="panel">
      <h3>${evaluation.label} <span class="muted">· ${evaluation.dataset} / ${evaluation.split}
        · ${summary.total} muestras</span></h3>
      <div class="row">
        <div class="stat"><b>${fmt(summary.accuracy)}</b><span>accuracy</span></div>
        <div class="stat"><b>${summary.correct}</b><span>aciertos</span></div>
        <div class="stat"><b>${summary.failed}</b><span>fallos</span></div>
        <div class="stat"><b>${summary.ambiguous}</b><span>ambiguas (margen &lt; ${summary.ambiguous_threshold})</span></div>
      </div>

      <h3>Matriz de confusión <span class="muted">(filas = etiqueta real, columnas = predicción; click en una celda para filtrar)</span></h3>
      <table class="grid confusion">
        <thead><tr><th></th>${head}</tr></thead>
        <tbody>${rows}</tbody>
      </table>

      <h3>Filtrar resultados</h3>
      <div class="row">
        <label>Resultado
          <select id="filter-outcome">
            <option value="all">todas</option>
            <option value="correct">aciertos</option>
            <option value="failed">fallos</option>
            <option value="ambiguous">ambiguas</option>
          </select>
        </label>
        <label>Etiqueta real
          <select id="filter-label"></select>
        </label>
        <label>Predicción
          <select id="filter-pred"></select>
        </label>
        <button id="btn-filter">Aplicar filtro</button>
        <button id="btn-save-ds" class="primary">Guardar filtro como dataset</button>
      </div>
      <p id="filter-info" class="hint"></p>
      <div id="results" class="samples"></div>
      <div class="row">
        <button id="btn-prev">← anteriores</button>
        <button id="btn-next">siguientes →</button>
      </div>
    </div>`;

  const classOptions = `<option value="">(cualquiera)</option>` +
    Array.from({ length: classes }, (_, i) => `<option value="${i}">${i}</option>`).join("");
  $("filter-label").innerHTML = classOptions;
  $("filter-pred").innerHTML = classOptions;
  $("filter-outcome").value = state.evalFilter.outcome;
  $("filter-label").value = state.evalFilter.label;
  $("filter-pred").value = state.evalFilter.pred;

  $("btn-filter").onclick = () => {
    state.evalFilter = {
      outcome: $("filter-outcome").value,
      label: $("filter-label").value,
      pred: $("filter-pred").value,
    };
    state.evalOffset = 0;
    loadResults();
  };
  $("btn-save-ds").onclick = saveFilterAsDataset;
  $("btn-prev").onclick = () => {
    state.evalOffset = Math.max(0, state.evalOffset - PAGE);
    loadResults();
  };
  $("btn-next").onclick = () => {
    state.evalOffset += PAGE;
    loadResults();
  };
  document.querySelectorAll("#eval-detail .cell").forEach((cell) => {
    cell.onclick = () => {
      state.evalFilter = {
        outcome: "all",
        label: cell.dataset.label,
        pred: cell.dataset.pred,
      };
      state.evalOffset = 0;
      $("filter-outcome").value = "all";
      $("filter-label").value = cell.dataset.label;
      $("filter-pred").value = cell.dataset.pred;
      loadResults();
    };
  });
}

function filterQuery() {
  const { outcome, label, pred } = state.evalFilter;
  const params = new URLSearchParams({
    outcome,
    offset: state.evalOffset,
    limit: PAGE,
  });
  if (label !== "") params.set("label", label);
  if (pred !== "") params.set("pred", pred);
  return params.toString();
}

async function loadResults() {
  const evaluation = state.currentEval;
  if (!evaluation) return;
  const payload = await api(`/api/evaluations/${evaluation.id}/results?${filterQuery()}`);

  const from = payload.total ? payload.offset + 1 : 0;
  const to = Math.min(payload.offset + PAGE, payload.total);
  $("filter-info").textContent =
    `${payload.total} muestra(s) cumplen el filtro — mostrando ${from}–${to}.`;
  $("btn-prev").disabled = payload.offset === 0;
  $("btn-next").disabled = to >= payload.total;

  const container = $("results");
  container.innerHTML = "";
  for (const result of payload.results) {
    const div = document.createElement("div");
    div.className = `sample ${result.correct ? "ok" : "bad"}`;
    div.innerHTML = `<img src="${result.thumb}" alt="${result.label}">
      <span>real ${result.label} · pred ${result.pred}</span>
      <span class="muted">m ${result.margin.toFixed(2)}${result.ambiguous ? " ⚠" : ""}</span>`;
    div.title = `#${result.index} · confianza ${result.conf.toFixed(3)}`;
    div.onclick = () => {
      goToTab("features");
      $("feat-exp").value = evaluation.experiment;
      $("feat-dataset").value = evaluation.dataset;
      $("feat-split").value = evaluation.split;
      predict(result.index);
    };
    container.appendChild(div);
  }
}

async function saveFilterAsDataset() {
  const evaluation = state.currentEval;
  const { outcome, label, pred } = state.evalFilter;
  const parts = [outcome === "all" ? "todas" : outcome];
  if (label !== "") parts.push(`real${label}`);
  if (pred !== "") parts.push(`pred${pred}`);
  const suggestion = `${parts.join("-")}-${evaluation.id.replace("eval_", "")}`;

  const name = prompt("Nombre del nuevo dataset", suggestion);
  if (!name) return;
  try {
    const definition = await api("/api/custom-datasets", {
      method: "POST",
      body: JSON.stringify({
        name,
        from_evaluation: { evaluation: evaluation.id, outcome, label, pred },
      }),
    });
    await refreshDatasets($("nn-select").value);
    alert(
      `Dataset «${definition.name}» creado con ${definition.size} muestras.\n` +
      "Ya aparece en Entrenar, Probar y Datasets."
    );
  } catch (error) {
    alert(error.message);
  }
}

// --- datasets (CRUD) --------------------------------------------------------

async function loadDatasets() {
  await refreshDatasets(null);
  const customs = await api("/api/custom-datasets");
  const byName = Object.fromEntries(customs.map((d) => [d.name, d]));

  fillSelect(
    $("ds-base"),
    state.datasets
      .filter((ds) => !ds.is_custom)
      .map((ds) => ({ value: ds.name, text: ds.label }))
  );

  const body = $("datasets-table").querySelector("tbody");
  body.innerHTML = "";
  for (const dataset of state.datasets) {
    const custom = byName[dataset.name];
    const users = custom
      ? [...custom.users.experiments, ...custom.users.evaluations]
      : [];
    const origin = custom?.origin?.type === "evaluation"
      ? `filtro ${JSON.stringify(custom.origin.filter)} de ${custom.origin.evaluation}`
      : custom
        ? `recorte de ${custom.origin.base}/${custom.origin.split}`
        : "builtin";

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${dataset.name}</td>
      <td>${dataset.is_custom ? "custom" : "builtin"}</td>
      <td>${custom ? `${custom.base} / ${custom.split}` : "—"}</td>
      <td>${custom ? custom.size : "—"}</td>
      <td class="origin">${origin}</td>
      <td>${users.length ? users.length : "—"}</td>
      <td class="actions"></td>`;
    const actions = row.querySelector(".actions");

    const view = document.createElement("button");
    view.className = "mini";
    view.textContent = "ver muestras";
    view.onclick = () => showDatasetSamples(dataset, custom);
    actions.append(view);

    if (custom) {
      const rename = document.createElement("button");
      rename.className = "mini";
      rename.textContent = "renombrar";
      rename.onclick = async () => {
        const name = prompt("Nuevo nombre", custom.name);
        if (!name) return;
        try {
          await api(`/api/custom-datasets/${encodeURIComponent(custom.name)}`, {
            method: "PATCH",
            body: JSON.stringify({ name }),
          });
          loadDatasets();
        } catch (error) {
          alert(error.message);
        }
      };

      const remove = document.createElement("button");
      remove.className = "mini";
      remove.textContent = "eliminar";
      remove.onclick = async () => {
        if (!confirm(`¿Eliminar el dataset «${custom.name}»?`)) return;
        try {
          await api(`/api/custom-datasets/${encodeURIComponent(custom.name)}`, {
            method: "DELETE",
          });
          loadDatasets();
        } catch (error) {
          alert(error.message);
        }
      };
      actions.append(rename, remove);
    }
    body.appendChild(row);
  }
}

async function showDatasetSamples(dataset, custom) {
  const split = custom ? "train" : "test";
  const payload = await api(
    `/api/datasets/${encodeURIComponent(dataset.name)}/samples?split=${split}&limit=24`
  );
  const container = $("ds-samples");
  container.innerHTML = `<p class="hint">«${dataset.name}» · ${payload.total} muestras
    (primeras ${payload.samples.length})</p>`;
  for (const sample of payload.samples) {
    const div = document.createElement("div");
    div.className = "sample";
    div.innerHTML = `<img src="${sample.thumb}" alt="${sample.label}">
      <span>#${sample.index} · ${sample.label}</span>`;
    container.appendChild(div);
  }
}

async function createDataset() {
  $("ds-error").textContent = "";
  const limit = $("ds-limit").value;
  try {
    const definition = await api("/api/custom-datasets", {
      method: "POST",
      body: JSON.stringify({
        name: $("ds-name").value,
        from_base: {
          base: $("ds-base").value,
          split: $("ds-split").value,
          limit: limit ? parseInt(limit, 10) : null,
        },
      }),
    });
    $("ds-name").value = "";
    await loadDatasets();
    $("ds-error").textContent = "";
    alert(`Dataset «${definition.name}» creado con ${definition.size} muestras.`);
  } catch (error) {
    $("ds-error").textContent = error.message;
  }
}

// --- features ---------------------------------------------------------------

async function refreshFeatureSelectors() {
  await refreshDatasets($("nn-select").value);
  fillSelect(
    $("feat-exp"),
    trainedExperiments().map((exp) => ({ value: exp.id, text: `${exp.label} · ${exp.id}` }))
  );
  fillSelect(
    $("feat-dataset"),
    state.datasets.map((ds) => ({ value: ds.name, text: ds.label }))
  );
  $("feat-error").textContent = trainedExperiments().length
    ? ""
    : "No hay ninguna NN entrenada todavía. Entrena una en la pestaña Entrenar.";
}

async function loadSamples() {
  $("feat-error").textContent = "";
  const dataset = $("feat-dataset").value;
  const split = $("feat-split").value;
  const offset = parseInt($("feat-offset").value, 10) || 0;
  try {
    const payload = await api(
      `/api/datasets/${encodeURIComponent(dataset)}/samples?split=${split}&offset=${offset}&limit=24`
    );
    const container = $("samples");
    container.innerHTML = "";
    for (const sample of payload.samples) {
      const div = document.createElement("div");
      div.className = "sample";
      div.innerHTML = `<img src="${sample.thumb}" alt="${sample.label}">
        <span>#${sample.index} · ${sample.label}</span>`;
      div.onclick = () => {
        document.querySelectorAll("#samples .sample").forEach((s) => s.classList.remove("selected"));
        div.classList.add("selected");
        predict(sample.index);
      };
      container.appendChild(div);
    }
  } catch (error) {
    $("feat-error").textContent = error.message;
  }
}

async function predict(index) {
  $("feat-error").textContent = "";
  try {
    const payload = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        experiment: $("feat-exp").value,
        dataset: $("feat-dataset").value,
        split: $("feat-split").value,
        index,
      }),
    });
    state.prediction = payload;
    renderPrediction(payload);
    renderLayers(payload.layers, "feature map");
    $("matrix-panel").innerHTML = "";
  } catch (error) {
    $("feat-error").textContent = error.message;
  }
}

async function loadKernels() {
  $("feat-error").textContent = "";
  try {
    const payload = await api(`/api/experiments/${$("feat-exp").value}/kernels`);
    $("prediction").innerHTML =
      `<p class="hint">Kernels aprendidos por cada capa (matrices de pesos).</p>`;
    renderLayers(payload.layers, "kernel");
    $("matrix-panel").innerHTML = "";
  } catch (error) {
    $("feat-error").textContent = error.message;
  }
}

function renderPrediction(payload) {
  const verdict = payload.correct
    ? `<span class="pred-ok">acierto</span>`
    : `<span class="pred-bad">fallo</span>`;
  const top = payload.top.map((t) => `${t.digit}: ${fmt(t.prob, 3)}`).join(" · ");
  $("prediction").innerHTML = `
    <div class="pred-box">
      <img src="${payload.thumb}" alt="muestra">
      <div>
        <div>Muestra <code>#${payload.index}</code> · etiqueta <b>${payload.label}</b>
          · predicción <b>${payload.pred}</b> ${verdict}</div>
        <div class="muted">top: ${top} · margen ${fmt(payload.margin, 3)}</div>
        <div class="muted">Click en cualquier mapa para ver su matriz de números.</div>
      </div>
    </div>`;
}

function drawMap(canvas, matrix, min, max) {
  const height = matrix.length;
  const width = matrix[0].length;
  const cell = Math.max(2, Math.round(72 / Math.max(height, width)));
  canvas.width = width * cell;
  canvas.height = height * cell;
  const ctx = canvas.getContext("2d");
  const range = max - min || 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const norm = (matrix[y][x] - min) / range;
      const shade = Math.round(255 * norm);
      ctx.fillStyle = `rgb(${shade}, ${Math.round(shade * 0.85)}, ${Math.round(60 + shade * 0.6)})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}

function renderLayers(layers, kind) {
  const container = $("layers");
  container.innerHTML = "";
  for (const layer of layers) {
    const div = document.createElement("div");
    div.className = "layer";
    const spec = layer.spec;
    const pool = spec.pool
      ? `${spec.pool.type}-pool ${spec.pool.size}x${spec.pool.size}`
      : "sin pooling";
    const size = kind === "kernel"
      ? `${layer.kernel_size}x${layer.kernel_size}`
      : `${layer.height}x${layer.width}`;
    div.innerHTML = `<h3>Capa ${layer.layer} · ${layer.kernels} kernels
      ${spec.kernel_size}x${spec.kernel_size} · ${spec.activation} · ${pool}
      <span class="muted">— ${layer.shown} ${kind}(s) de ${size}${
        layer.truncated ? " (truncado)" : ""
      }</span></h3>`;
    const maps = document.createElement("div");
    maps.className = "maps";

    for (const map of layer.maps) {
      const item = document.createElement("div");
      item.className = "map";
      const canvas = document.createElement("canvas");
      drawMap(canvas, map.matrix, map.min, map.max);
      const caption = document.createElement("span");
      caption.textContent = `k${map.index}`;
      item.append(canvas, caption);
      item.onclick = () => {
        document.querySelectorAll(".map").forEach((m) => m.classList.remove("selected"));
        item.classList.add("selected");
        renderMatrix(layer, map, kind);
      };
      maps.appendChild(item);
    }
    div.appendChild(maps);
    container.appendChild(div);
  }
}

function renderMatrix(layer, map, kind) {
  const range = (map.max - map.min) || 1;
  const rows = map.matrix
    .map(
      (row) =>
        `<tr>${row
          .map((value) => {
            const norm = (value - map.min) / range;
            const shade = Math.round(40 + 90 * norm);
            return `<td style="background: rgb(${shade}, ${shade + 8}, ${shade + 20})">${value.toFixed(2)}</td>`;
          })
          .join("")}</tr>`
    )
    .join("");
  $("matrix-panel").innerHTML = `
    <div class="matrix-wrap">
      <h3>Representación matricial · capa ${layer.layer} · ${kind} k${map.index}
        <span class="muted">(${map.matrix.length}x${map.matrix[0].length},
        min ${fmt(map.min, 3)}, max ${fmt(map.max, 3)}, media ${fmt(map.mean, 3)})</span></h3>
      <table class="matrix">${rows}</table>
    </div>`;
  $("matrix-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// --- arranque ---------------------------------------------------------------

$("nn-select").addEventListener("change", onNnChange);
$("dataset-select").addEventListener("change", onDatasetChange);
$("btn-train").addEventListener("click", startTraining);
$("btn-stop").addEventListener("click", stopTraining);
$("btn-reset").addEventListener("click", onNnChange);
$("btn-add-layer").addEventListener("click", addLayer);
$("btn-remove-layer").addEventListener("click", removeLayer);
$("eval-exp").addEventListener("change", onEvalExperimentChange);
$("eval-dataset").addEventListener("change", onEvalDatasetChange);
$("btn-evaluate").addEventListener("click", evaluate);
$("btn-create-ds").addEventListener("click", createDataset);
$("btn-samples").addEventListener("click", loadSamples);
$("btn-kernels").addEventListener("click", loadKernels);

loadCatalogs().catch((error) => {
  $("train-error").textContent = error.message;
});
