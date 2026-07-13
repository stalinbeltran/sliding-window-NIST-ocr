"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  nns: [],
  datasets: [],
  experiments: [],
  training: null,      // { id, since, epochs: [], batches: [] }
  poll: null,
  prediction: null,
  selectedMap: null,   // { layer, index, kind }
};

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

function readConfig() {
  return JSON.parse($("config-json").value);
}

function writeConfig(config) {
  $("config-json").value = JSON.stringify(config, null, 2);
}

const fmt = (value, digits = 4) =>
  typeof value === "number" ? value.toFixed(digits) : "—";

function describeArchitecture(config) {
  const layers = config?.model?.layers || [];
  return layers
    .map((layer) => {
      const pool = layer.pool ? `+${layer.pool.type}${layer.pool.size}` : "";
      return `${layer.kernels}@${layer.kernel_size}x${layer.kernel_size}${pool}`;
    })
    .join(" → ");
}

// --- pestañas ---------------------------------------------------------------

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    button.classList.add("active");
    $(`tab-${button.dataset.tab}`).classList.add("active");
    if (button.dataset.tab === "experimentos") loadExperiments();
    if (button.dataset.tab === "features") refreshFeatureSelectors();
  });
});

// --- catálogos --------------------------------------------------------------

async function loadCatalogs() {
  state.nns = await api("/api/nns");
  const nnSelect = $("nn-select");
  nnSelect.innerHTML = state.nns
    .map((nn) => `<option value="${nn.name}">${nn.label}</option>`)
    .join("");
  await onNnChange();
  await loadExperiments();
}

async function onNnChange() {
  const nn = state.nns.find((item) => item.name === $("nn-select").value);
  $("nn-desc").textContent = nn.description;
  state.datasets = await api(`/api/datasets?nn=${nn.name}`);
  $("dataset-select").innerHTML = state.datasets
    .map((ds) => `<option value="${ds.name}">${ds.label}</option>`)
    .join("");
  writeConfig(structuredClone(nn.default_config));
  $("dataset-select").value = nn.default_config.dataset.name;
}

// Al cambiar de dataset se reemplazan los params por SUS defaults (regla 12).
function onDatasetChange() {
  const config = readConfig();
  const dataset = state.datasets.find((ds) => ds.name === $("dataset-select").value);
  config.dataset = { name: dataset.name, params: structuredClone(dataset.defaults) };
  writeConfig(config);
}

// --- editor de arquitectura -------------------------------------------------

function addLayer() {
  const config = readConfig();
  const last = config.model.layers[config.model.layers.length - 1];
  config.model.layers.push(
    structuredClone(last) || {
      kernels: 8,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      activation: "relu",
      pool: null,
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

  const running = payload.running;
  $("live-header").innerHTML = running
    ? `<b>${id}</b> · corriendo · ${state.training.epochs.length} época(s)`
    : `<b>${id}</b> · ${payload.status}`;

  if (running) {
    state.poll = setTimeout(pollMetrics, 1000);
  } else {
    $("btn-stop").disabled = true;
    const experiment = await api(`/api/experiments/${id}`);
    if (experiment.final_test) {
      $("live-header").innerHTML +=
        ` · test acc <b>${fmt(experiment.final_test.acc)}</b>`;
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
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!state.training) return;

  const batches = state.training.batches;
  const epochs = state.training.epochs;
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

  plot(batches.map((b) => Math.min(b.loss / 2.5, 1)), "#ff9f43"); // loss por lote (escalada)
  plot(batches.map((b) => b.acc), "#4da3ff");                     // acc por lote
  plot(epochs.map((e) => e.val_acc), "#3ddc84");                  // val acc por época

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

function renderInitFrom() {
  const trained = state.experiments.filter((exp) => exp.has_checkpoint);
  $("initfrom-select").innerHTML =
    `<option value="">(desde cero)</option>` +
    trained
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
      <td>${fmt(exp.best?.val_acc)}</td>
      <td>${fmt(exp.final_test?.acc)}</td>
      <td class="actions"></td>`;
    const actions = row.querySelector(".actions");

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
      document.querySelector('.tab[data-tab="entrenar"]').click();
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

    actions.append(rename, duplicate, retrain, remove);
    body.appendChild(row);
  }
}

// --- features ---------------------------------------------------------------

function refreshFeatureSelectors() {
  const trained = state.experiments.filter((exp) => exp.has_checkpoint);
  $("feat-exp").innerHTML = trained
    .map((exp) => `<option value="${exp.id}">${exp.label} · ${exp.id}</option>`)
    .join("");
  $("feat-dataset").innerHTML = state.datasets
    .map((ds) => `<option value="${ds.name}">${ds.label}</option>`)
    .join("");
  if (!trained.length) {
    $("feat-error").textContent =
      "No hay ninguna NN entrenada todavía. Entrena una en la pestaña Entrenar.";
  } else {
    $("feat-error").textContent = "";
  }
}

async function loadSamples() {
  $("feat-error").textContent = "";
  const dataset = $("feat-dataset").value;
  const split = $("feat-split").value;
  const offset = parseInt($("feat-offset").value, 10) || 0;
  try {
    const payload = await api(
      `/api/datasets/${dataset}/samples?split=${split}&offset=${offset}&limit=24`
    );
    const container = $("samples");
    container.innerHTML = "";
    for (const sample of payload.samples) {
      const div = document.createElement("div");
      div.className = "sample";
      div.innerHTML = `<img src="${sample.thumb}" alt="${sample.label}">
        <span>#${sample.index} · ${sample.label}</span>`;
      div.onclick = () => {
        document.querySelectorAll(".sample").forEach((s) => s.classList.remove("selected"));
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
  const top = payload.top
    .map((t) => `${t.digit}: ${fmt(t.prob, 3)}`)
    .join(" · ");
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
    const pool = spec.pool ? `${spec.pool.type}-pool ${spec.pool.size}x${spec.pool.size}` : "sin pooling";
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
  const rows = map.matrix
    .map(
      (row) =>
        `<tr>${row
          .map((value) => {
            const range = (map.max - map.min) || 1;
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
$("btn-samples").addEventListener("click", loadSamples);
$("btn-kernels").addEventListener("click", loadKernels);

loadCatalogs().catch((error) => {
  $("train-error").textContent = error.message;
});
