// Frontend: entrenamiento (con re-entrenamiento), pruebas de NNs entrenadas
// (evaluaciones, miniaturas, filtros, animación de la ventana deslizante) y
// CRUD de experimentos y datasets custom.

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => {
  if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || r.statusText); });
  return r.json();
});
const post = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});
const patch = (path, body) => api(path, {
  method: "PATCH", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const del = (path) => api(path, { method: "DELETE" });

let NNS = [];
let currentDetail = null;

// ---------- gráficas (canvas, sin librerías) ----------

const COLORS = ["#2f6fed", "#d64545", "#1a7f37", "#b57d00"];
const GRAY = "#c9d1de";

function drawChart(canvas, series, opts = {}) {
  // series: [{label|null, points: [{x, y}], color?, width?}] — label null = sin leyenda
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const m = { left: 48, right: 12, top: 10, bottom: 28 };
  ctx.clearRect(0, 0, W, H);

  const allPts = series.flatMap((s) => s.points);
  if (!allPts.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "13px sans-serif";
    ctx.fillText("Sin datos todavía…", 20, H / 2);
    return;
  }
  let xMin = Math.min(...allPts.map((p) => p.x)), xMax = Math.max(...allPts.map((p) => p.x));
  let yMin = Math.min(...allPts.map((p) => p.y)), yMax = Math.max(...allPts.map((p) => p.y));
  if (opts.yMin !== undefined) yMin = opts.yMin;
  if (opts.yMax !== undefined) yMax = opts.yMax;
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) yMax = yMin + 1e-6;

  const sx = (x) => m.left + ((x - xMin) / (xMax - xMin)) * (W - m.left - m.right);
  const sy = (y) => H - m.bottom - ((y - yMin) / (yMax - yMin)) * (H - m.top - m.bottom);

  // ejes y grilla
  ctx.strokeStyle = "#dde2ea";
  ctx.fillStyle = "#667085";
  ctx.font = "11px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + (i / 4) * (yMax - yMin);
    const y = sy(yv);
    ctx.beginPath(); ctx.moveTo(m.left, y); ctx.lineTo(W - m.right, y); ctx.stroke();
    ctx.fillText(yv.toPrecision(3), 4, y + 4);
  }
  for (let i = 0; i <= 4; i++) {
    const xv = xMin + (i / 4) * (xMax - xMin);
    ctx.fillText(Number.isInteger(xMax) ? Math.round(xv) : xv.toPrecision(3), sx(xv) - 6, H - 8);
  }

  let legendX = m.left + 8;
  series.forEach((s, i) => {
    const color = s.color || COLORS[i % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = s.width || 1.8;
    ctx.beginPath();
    s.points.forEach((p, j) => (j ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
    ctx.stroke();
    if (s.label) {
      ctx.fillStyle = color;
      ctx.fillRect(legendX, m.top, 10, 10);
      ctx.fillStyle = "#1c2330";
      ctx.fillText(s.label, legendX + 14, m.top + 9);
      legendX += 14 + ctx.measureText(s.label).width + 16;
    }
  });
}

// ---------- pestañas ----------

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab-page").forEach((p) => (p.hidden = p.id !== "tab-" + b.dataset.tab));
  if (b.dataset.tab === "probar") initTestTab();
  if (b.dataset.tab === "datasets") refreshDatasetsTab();
}));

// ============================================================
// PESTAÑA ENTRENAR
// ============================================================

async function loadNNs() {
  NNS = await api("/api/nns");
  const sel = $("nn-select");
  sel.innerHTML = NNS.map((n) => `<option value="${n.name}">${n.name}</option>`).join("");
  await onNNChange();
}

let DATASETS = []; // datasets compatibles con la NN seleccionada (con sus defaults)
let TRAIN_EXPS = []; // experimentos de la NN seleccionada (para init_from)

function effectiveWindow(dsEntry, params) {
  // Los defaults del catálogo ya incluyen window_size (los custom heredan los del base).
  const merged = { ...dsEntry.defaults, ...params };
  return merged.window_size ?? 28;
}

async function onNNChange() {
  const nn = $("nn-select").value;
  const spec = NNS.find((n) => n.name === nn);
  $("nn-description").textContent = spec.description;

  DATASETS = await api(`/api/datasets?nn=${nn}`);
  const dsel = $("dataset-select");
  dsel.innerHTML = DATASETS.map((d) =>
    `<option value="${d.name}">${d.name}${d.custom ? " (custom, " + d.count + ")" : ""}</option>`).join("");

  const config = JSON.parse(JSON.stringify(spec.defaults));
  const isSeq = nn === "secuenciador";
  $("dim-exp-row").hidden = !isSeq;
  if (isSeq) {
    const dims = await api("/api/experiments?nn=dimensionador&status=completed");
    const dsel2 = $("dim-exp-select");
    dsel2.innerHTML = dims.length
      ? dims.map((e) => `<option value="${e.id}">${expName(e)} (val_acc=${e.status.best?.val_acc ?? "?"})</option>`).join("")
      : `<option value="">— entrena primero un dimensionador —</option>`;
    config.dimensionador_experiment = dsel2.value || null;
  }

  // Re-entrenamiento: experimentos de esta NN con checkpoint utilizable
  const all = await api(`/api/experiments?nn=${nn}`);
  TRAIN_EXPS = all.filter((e) => e.status.best);
  $("init-from-select").innerHTML = `<option value="">— empezar de cero —</option>` +
    TRAIN_EXPS.map((e) => `<option value="${e.id}">${expName(e)}</option>`).join("");

  // Preseleccionar en el desplegable el dataset que trae la config por defecto,
  // para que ambos queden siempre en sincronía.
  if (config.dataset && DATASETS.some((d) => d.name === config.dataset.name)) {
    dsel.value = config.dataset.name;
  }
  $("config-json").value = JSON.stringify(config, null, 2);
  onDatasetChange();
}

function onDatasetChange() {
  const name = $("dataset-select").value;
  const spec = DATASETS.find((d) => d.name === name);
  $("dataset-description").textContent = spec ? spec.description : "";
  if (!spec) return;
  try {
    const config = JSON.parse($("config-json").value || "{}");
    if (config.dataset) {
      // Sincronizar nombre Y parámetros: cada dataset acepta parámetros distintos.
      config.dataset.name = spec.name;
      config.dataset.params = JSON.parse(JSON.stringify(spec.defaults));
      // El dimensionador debe declarar la ventana con la que realmente entrena.
      if ($("nn-select").value === "dimensionador" && config.model && !config.init_from) {
        config.model.window_size = effectiveWindow(spec, config.dataset.params);
      }
      $("config-json").value = JSON.stringify(config, null, 2);
    }
  } catch { /* config siendo editada a mano */ }
}

function onInitFromChange() {
  const initId = $("init-from-select").value;
  try {
    const config = JSON.parse($("config-json").value || "{}");
    if (initId) {
      const exp = TRAIN_EXPS.find((e) => e.id === initId);
      config.init_from = initId;
      if (exp) {
        config.model = JSON.parse(JSON.stringify(exp.config.model));
        // Secuenciador: por defecto se continúa con la misma trayectoria (mismo
        // stride) que el experimento origen; editable en la config si se quiere
        // re-entrenar con otra.
        const origStride = exp.config.dataset?.params?.stride;
        if ($("nn-select").value === "secuenciador" && origStride !== undefined &&
            config.dataset?.params) {
          config.dataset.params.stride = origStride;
        }
      }
    } else {
      delete config.init_from;
    }
    $("config-json").value = JSON.stringify(config, null, 2);
  } catch { /* config siendo editada a mano */ }
}

async function startTraining() {
  const msg = $("train-msg");
  msg.textContent = "";
  let config;
  try { config = JSON.parse($("config-json").value); }
  catch (e) { msg.textContent = "JSON inválido: " + e.message; return; }
  const nn = $("nn-select").value;
  // La config JSON es la fuente de verdad (el desplegable ya la mantiene en sincronía).
  if (nn === "secuenciador") config.dimensionador_experiment = $("dim-exp-select").value || null;
  try {
    const res = await post("/api/train", { nn, config });
    msg.textContent = "Entrenamiento iniciado: " + res.experiment_id;
    await refreshExperiments();
    showDetail(res.experiment_id);
  } catch (e) { msg.textContent = "Error: " + e.message; }
}

// ---------- listado de experimentos (con CRUD) ----------

function expName(e) {
  return e.status.label ? `${e.status.label} (${e.id})` : e.id;
}

async function refreshExperiments() {
  const exps = await api("/api/experiments");
  const tbody = $("exp-table").querySelector("tbody");
  tbody.innerHTML = "";
  exps.slice().reverse().forEach((e) => {
    const st = e.status.status;
    const tr = document.createElement("tr");
    tr.className = "selectable";
    tr.innerHTML = `
      <td>${e.status.label ? `<b>${e.status.label}</b><br><span class="hint">${e.id}</span>` : e.id}</td>
      <td class="status-${st}">${st}${e.running ? " ⏳" : ""}</td>
      <td>${e.status.best?.val_acc ?? "—"}</td>
      <td>${e.status.final_test?.acc ?? "—"}</td>
      <td class="actions">
        ${e.running ? `<button class="small danger" data-stop="${e.id}">Detener</button>` : `
        <button class="small secondary" data-rename="${e.id}" title="Renombrar (alias)">✎</button>
        <button class="small secondary" data-copy="${e.id}" title="Duplicar">⧉</button>
        <button class="small danger" data-del="${e.id}" title="Eliminar">🗑</button>`}
      </td>`;
    tr.addEventListener("click", (ev) => {
      if (ev.target.closest("button")) return;
      showDetail(e.id);
    });
    tbody.appendChild(tr);
  });
  const msg = $("exp-msg");
  tbody.querySelectorAll("[data-stop]").forEach((b) =>
    b.addEventListener("click", () => api(`/api/experiments/${b.dataset.stop}/stop`, { method: "POST" })));
  tbody.querySelectorAll("[data-rename]").forEach((b) =>
    b.addEventListener("click", async () => {
      const name = prompt("Nombre visible del experimento (el id no cambia porque otros lo referencian):");
      if (name === null) return;
      try { await patch(`/api/experiments/${b.dataset.rename}`, { label: name }); await refreshExperiments(); }
      catch (e) { msg.textContent = "Error: " + e.message; }
    }));
  tbody.querySelectorAll("[data-copy]").forEach((b) =>
    b.addEventListener("click", async () => {
      try {
        const r = await post(`/api/experiments/${b.dataset.copy}/copy`);
        msg.textContent = "Duplicado como " + r.experiment_id;
        await refreshExperiments();
      } catch (e) { msg.textContent = "Error: " + e.message; }
    }));
  tbody.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`¿Eliminar ${b.dataset.del}? (queda el respaldo en backups/)`)) return;
      try { await del(`/api/experiments/${b.dataset.del}`); msg.textContent = "Eliminado."; await refreshExperiments(); }
      catch (e) { msg.textContent = "Error: " + e.message; }
    }));
}

// ---------- detalle + métricas en vivo ----------

function showDetail(expId) {
  currentDetail = expId;
  $("panel-detail").hidden = false;
  $("detail-title").textContent = expId;
  refreshDetail();
}

async function refreshDetail() {
  if (!currentDetail) return;
  const [info, metrics] = await Promise.all([
    api(`/api/experiments/${currentDetail}`),
    api(`/api/experiments/${currentDetail}/metrics`),
  ]);
  const s = info.status;
  $("detail-status").innerHTML =
    `Estado: <span class="status-${s.status}">${s.status}</span>` +
    (s.best ? ` · mejor val_acc=${s.best.val_acc} (época ${s.best.epoch})` : "") +
    (s.final_test ? ` · test acc=${s.final_test.acc}` : "") +
    (s.error ? `<pre>${s.error}</pre>` : "");
  $("detail-config").textContent = JSON.stringify(info.config, null, 2);

  const epochs = metrics.filter((m) => m.type === "epoch");
  const batches = metrics.filter((m) => m.type === "batch");
  const lastEpoch = batches.length ? batches[batches.length - 1].epoch : 1;
  const curBatches = batches.filter((b) => b.epoch === lastEpoch);

  drawChart($("chart-loss"), [
    { label: "train_loss", points: epochs.map((e) => ({ x: e.epoch, y: e.train_loss })) },
    { label: "val_loss", points: epochs.map((e) => ({ x: e.epoch, y: e.val_loss })) },
  ]);
  drawChart($("chart-acc"), [
    { label: "train_acc", points: epochs.map((e) => ({ x: e.epoch, y: e.train_acc })) },
    { label: "val_acc", points: epochs.map((e) => ({ x: e.epoch, y: e.val_acc })) },
  ], { yMax: 1 });
  drawChart($("chart-batch"), [
    { label: `loss (época ${lastEpoch})`, points: curBatches.map((b) => ({ x: b.step, y: b.loss })) },
  ]);

  const lastPerStep = [...epochs].reverse().find((e) => e.val_per_step_acc);
  $("per-step-box").hidden = !lastPerStep;
  if (lastPerStep) {
    drawChart($("chart-step"), [
      { label: "val acc por paso", points: lastPerStep.val_per_step_acc.map((a, i) => ({ x: i + 1, y: a })) },
    ], { yMin: 0, yMax: 1 });
  }
}

// ============================================================
// PESTAÑA PROBAR
// ============================================================

let TEST_EXPS = [];      // experimentos con checkpoint utilizable
let TEST_DS = [];        // datasets compatibles con el experimento elegido
let RESULTS = null;      // {kind: "eval"|"browse", evalId?, dsName, params, split}
let PAGE = { offset: 0, limit: 60, total: 0 };
let evalWasRunning = {};

async function initTestTab() {
  const all = await api("/api/experiments");
  TEST_EXPS = all.filter((e) => e.status.best);
  const sel = $("test-exp-select");
  const prev = sel.value;
  sel.innerHTML = TEST_EXPS.length
    ? TEST_EXPS.map((e) => `<option value="${e.id}">${expName(e)} [${e.config.nn}]</option>`).join("")
    : `<option value="">— entrena primero una NN —</option>`;
  if (prev && TEST_EXPS.some((e) => e.id === prev)) sel.value = prev;
  await onTestExpChange();
  await refreshEvals();
}

function testExp() { return TEST_EXPS.find((e) => e.id === $("test-exp-select").value); }

async function onTestExpChange() {
  const exp = testExp();
  if (!exp) { $("test-dataset-select").innerHTML = ""; return; }
  TEST_DS = await api(`/api/datasets?nn=${exp.config.nn}`);
  const sel = $("test-dataset-select");
  sel.innerHTML = TEST_DS.map((d) =>
    `<option value="${d.name}">${d.name}${d.custom ? " (custom, " + d.count + ")" : ""}</option>`).join("");
  onTestDatasetChange();
}

function testDatasetParams() {
  // Params efectivos: defaults del dataset + los del entrenamiento del
  // experimento. La CNN se evalúa con la ventana con la que se entrenó; el
  // secuenciador además con su stride (la trayectoria es entrada de la red:
  // el servidor rechaza con 400 cualquier otro valor).
  const exp = testExp();
  const ds = TEST_DS.find((d) => d.name === $("test-dataset-select").value);
  if (!ds) return {};
  const params = JSON.parse(JSON.stringify(ds.defaults));
  if (exp && exp.config.nn === "dimensionador" && "window_size" in params) {
    params.window_size = exp.config.model.window_size;
  }
  if (exp && exp.config.nn === "secuenciador") {
    const tp = exp.config.dataset?.params || {};
    // Si el experimento (antiguo) no registró el valor, se omite y el servidor
    // fija el efectivo del entrenamiento en la config de la evaluación.
    if (tp.window_size !== undefined) params.window_size = tp.window_size;
    else delete params.window_size;
    if (tp.stride !== undefined) params.stride = tp.stride;
    else delete params.stride;
  }
  return params;
}

function onTestDatasetChange() {
  const ds = TEST_DS.find((d) => d.name === $("test-dataset-select").value);
  if (!ds) { $("test-dataset-description").textContent = ""; return; }
  const params = testDatasetParams();
  $("test-dataset-description").textContent =
    ds.description + (Object.keys(params).length ? "  ·  params: " + JSON.stringify(params) : "");
}

async function startEvaluation() {
  const msg = $("test-msg");
  msg.textContent = "";
  const exp = testExp();
  if (!exp) { msg.textContent = "No hay experimento seleccionado."; return; }
  const limit = $("test-limit").value ? parseInt($("test-limit").value, 10) : null;
  try {
    const r = await post("/api/evaluations", {
      experiment: exp.id,
      dataset: { name: $("test-dataset-select").value, params: testDatasetParams() },
      split: $("test-split").value,
      limit,
    });
    msg.textContent = "Evaluación iniciada: " + r.evaluation_id;
    await refreshEvals();
    openEvaluation(r.evaluation_id);
  } catch (e) { msg.textContent = "Error: " + e.message; }
}

function browseSamples() {
  const dsName = $("test-dataset-select").value;
  if (!dsName) return;
  RESULTS = { kind: "browse", dsName, params: testDatasetParams(), split: $("test-split").value };
  PAGE = { offset: 0, limit: 60, total: 0 };
  $("panel-results").hidden = false;
  $("results-title").textContent = `Muestras de ${dsName} (${RESULTS.split})`;
  $("results-summary").textContent = "Explorando el dataset sin evaluar: haz click en una muestra "
    + "para aplicarla a la NN seleccionada. Los filtros de acierto/ambigüedad requieren una evaluación.";
  loadResultsPage();
}

async function refreshEvals() {
  const evals = await api("/api/evaluations");
  const tbody = $("eval-table").querySelector("tbody");
  tbody.innerHTML = "";
  evals.slice().reverse().forEach((ev) => {
    const st = ev.status.status;
    const prog = ev.status.progress;
    const acc = ev.status.summary?.acc;
    const tr = document.createElement("tr");
    tr.className = "selectable";
    tr.innerHTML = `
      <td>${ev.id}</td>
      <td class="hint">${ev.config.experiment}</td>
      <td class="hint">${ev.config.dataset.name} (${ev.config.split}${ev.config.limit ? ", n≤" + ev.config.limit : ""})${
        ev.config.dataset.params?.window_size !== undefined ? `<br>ventana ${ev.config.dataset.params.window_size}` : ""}${
        ev.config.dataset.params?.stride !== undefined ? ` · stride ${ev.config.dataset.params.stride}` : ""}</td>
      <td class="status-${st}">${st}${ev.running ? ` ⏳ ${prog ? prog.done + "/" + prog.total : ""}` : ""}</td>
      <td>${acc ?? "—"}</td>
      <td>
        ${ev.running ? `<button class="small danger" data-stop="${ev.id}">Detener</button>`
                     : `<button class="small danger" data-del="${ev.id}">🗑</button>`}
      </td>`;
    tr.addEventListener("click", (e2) => {
      if (e2.target.closest("button")) return;
      openEvaluation(ev.id);
    });
    tbody.appendChild(tr);

    // recargar el grid cuando la evaluación seleccionada termina
    if (evalWasRunning[ev.id] && !ev.running && RESULTS?.evalId === ev.id) loadResultsPage();
    evalWasRunning[ev.id] = ev.running;
  });
  tbody.querySelectorAll("[data-stop]").forEach((b) =>
    b.addEventListener("click", () => post(`/api/evaluations/${b.dataset.stop}/stop`)));
  tbody.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`¿Eliminar ${b.dataset.del}?`)) return;
      try {
        await del(`/api/evaluations/${b.dataset.del}`);
        if (RESULTS?.evalId === b.dataset.del) { RESULTS = null; $("panel-results").hidden = true; }
        await refreshEvals();
      } catch (e) { $("test-msg").textContent = "Error: " + e.message; }
    }));
}

async function openEvaluation(evalId) {
  const info = await api(`/api/evaluations/${evalId}`);
  RESULTS = {
    kind: "eval", evalId,
    dsName: info.config.dataset.name,
    params: info.config.dataset.params || {},
    split: info.config.split,
    experiment: info.config.experiment,
  };
  PAGE = { offset: 0, limit: 60, total: 0 };
  $("panel-results").hidden = false;
  $("results-title").textContent = `Resultados de ${evalId}`;
  loadResultsPage();
}

function currentFilters() {
  return {
    result: $("filter-result").value,
    ambiguity: parseFloat($("filter-ambiguity").value) || 0.2,
    label: $("filter-label").value,
    pred: $("filter-pred").value,
  };
}

async function loadResultsPage() {
  if (!RESULTS) return;
  const grid = $("samples-grid");
  grid.innerHTML = `<p class="hint">Cargando…</p>`;
  try {
    if (RESULTS.kind === "eval") {
      const f = currentFilters();
      const q = new URLSearchParams({
        result: f.result, ambiguity: f.ambiguity,
        offset: PAGE.offset, limit: PAGE.limit,
      });
      if (f.label !== "") q.set("label", f.label);
      if (f.pred !== "") q.set("pred", f.pred);
      const r = await api(`/api/evaluations/${RESULTS.evalId}/results?` + q);
      PAGE.total = r.total;
      renderSummary(r.summary, r.total);
      renderGrid(r.items.map((it) => ({
        index: it.index, label: it.label, png: it.png,
        pred: it.pred, correct: it.correct, margin: it.margin,
      })));
    } else {
      const f = currentFilters();
      const q = new URLSearchParams({
        split: RESULTS.split, offset: PAGE.offset, limit: PAGE.limit,
        params: JSON.stringify(RESULTS.params),
      });
      if (f.label !== "") q.set("label", f.label);
      const r = await api(`/api/datasets/${RESULTS.dsName}/samples?` + q);
      PAGE.total = r.total;
      renderGrid(r.items);
    }
  } catch (e) {
    grid.innerHTML = `<p class="hint">Error: ${e.message}</p>`;
  }
  $("grid-page").textContent =
    `${PAGE.total ? PAGE.offset + 1 : 0}–${Math.min(PAGE.offset + PAGE.limit, PAGE.total)} de ${PAGE.total}`;
}

function renderSummary(summary, totalFiltered) {
  if (!summary) { $("results-summary").textContent = "Evaluación en curso…"; return; }
  $("results-summary").textContent =
    `n=${summary.n} · acc=${summary.acc} · fallos=${summary.errors}` +
    (summary.per_step_acc ? ` · acc por paso: [${summary.per_step_acc.join(", ")}]` : "") +
    ` · filtro actual: ${totalFiltered} muestras`;
}

function renderGrid(items) {
  const grid = $("samples-grid");
  grid.innerHTML = "";
  if (!items.length) {
    grid.innerHTML = `<p class="hint">Sin muestras con este filtro.</p>`;
    return;
  }
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "cell" + (it.correct === undefined ? "" : it.correct ? " ok" : " fail");
    div.innerHTML = `
      <img src="data:image/png;base64,${it.png}" alt="muestra ${it.index}">
      <span class="tag">#${it.index} · ${it.label}${it.pred !== undefined ? "→" + it.pred : ""}</span>`;
    div.addEventListener("click", () => {
      grid.querySelectorAll(".cell.selected").forEach((c) => c.classList.remove("selected"));
      div.classList.add("selected");
      openSample(it.index);
    });
    grid.appendChild(div);
  });
}

async function saveFilteredDataset() {
  const msg = $("save-ds-msg");
  msg.textContent = "";
  if (!RESULTS || RESULTS.kind !== "eval") {
    msg.textContent = "Primero abre una evaluación: el filtro se guarda sobre sus resultados.";
    return;
  }
  const f = currentFilters();
  try {
    const r = await post(`/api/evaluations/${RESULTS.evalId}/save-dataset`, {
      result: f.result, ambiguity: f.ambiguity,
      label: f.label === "" ? null : parseInt(f.label, 10),
      pred: f.pred === "" ? null : parseInt(f.pred, 10),
      name: $("save-ds-name").value || null,
    });
    msg.textContent = `Dataset guardado: ${r.name} (${r.count} muestras). Disponible en Entrenar y Probar.`;
    $("save-ds-name").value = "";
  } catch (e) { msg.textContent = "Error: " + e.message; }
}

// ---------- detalle de muestra: predicción + animación ----------

let SAMPLE = null;   // respuesta de /api/predict (+ imgEl cargada)
let animIdx = 0;
let animTimer = null;

async function openSample(index) {
  const exp = RESULTS.kind === "eval"
    ? TEST_EXPS.find((e) => e.id === RESULTS.experiment) || testExp()
    : testExp();
  if (!exp) return;
  stopAnim();
  $("panel-sample").hidden = false;
  $("sample-title").textContent = `#${index} — cargando…`;
  try {
    const r = await post("/api/predict", {
      experiment: exp.id,
      dataset: { name: RESULTS.dsName, params: RESULTS.params },
      split: RESULTS.split,
      index,
    });
    SAMPLE = r;
    const img = new Image();
    img.onload = () => { SAMPLE.imgEl = img; showSample(); };
    img.src = "data:image/png;base64," + r.image_png;
  } catch (e) {
    $("sample-title").textContent = `#${index}`;
    $("sample-verdict").textContent = "Error: " + e.message;
    $("sample-verdict").className = "verdict-fail";
  }
}

function showSample() {
  const r = SAMPLE;
  $("sample-title").textContent =
    `#${r.index} — ${r.nn} ${r.experiment}`;
  const ok = r.pred === r.label;
  $("sample-verdict").innerHTML =
    `Etiqueta <b>${r.label}</b> · predicción <b>${r.pred}</b> ` +
    `(conf ${r.conf}, margen ${r.margin}) — ` +
    (ok ? `<span class="verdict-ok">acierto</span>` : `<span class="verdict-fail">fallo</span>`) +
    (r.margin < 0.2 ? ` · <b>ambigua</b>` : "");
  $("trace-reason").textContent = r.trace ? "" : (r.trace_reason || "");
  $("anim-controls").hidden = !r.trace;
  $("trace-chart-box").hidden = !r.trace;
  animIdx = r.trace ? r.trace.steps.length - 1 : 0;
  renderSampleStep();
  if (r.trace) {
    drawTraceChart();
    startAnim(); // animación automática al abrir
  }
}

function renderSampleStep() {
  const r = SAMPLE;
  const canvas = $("sample-canvas");
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scale = canvas.width / r.image_size;
  ctx.drawImage(r.imgEl, 0, 0, canvas.width, canvas.height);

  let probs = r.probs, stepPred = r.pred;
  if (r.trace) {
    const t = r.trace, s = t.steps[animIdx];
    probs = s.probs; stepPred = s.pred;
    if (t.positions.length > animIdx) {
      const [top, left] = t.positions[animIdx];
      ctx.strokeStyle = "#2f6fed";
      ctx.lineWidth = 3;
      ctx.strokeRect(left * scale, top * scale, t.window_size * scale, t.window_size * scale);
    }
    $("anim-step").textContent =
      `paso ${animIdx + 1}/${t.steps.length} · ventana ${t.window_size}` +
      (t.stride ? ` · stride ${t.stride}` : "") + ` · salida: ${stepPred}`;
  }
  renderProbBars(probs, r.label, r.pred);
}

function renderProbBars(probs, label, finalPred) {
  const box = $("probs-bars");
  box.innerHTML = "";
  probs.forEach((p, d) => {
    const row = document.createElement("div");
    row.className = "bar-row" +
      (d === label ? " is-label" : d === finalPred && finalPred !== label ? " is-pred-wrong" : "");
    row.innerHTML = `
      <span class="digit">${d}</span>
      <div class="bar-bg"><div class="bar" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <span class="val">${p.toFixed(3)}</span>`;
    box.appendChild(row);
  });
}

function drawTraceChart() {
  const r = SAMPLE, t = r.trace;
  const T = t.steps.length;
  const seriesFor = (cls) => t.steps.map((s, i) => ({ x: i + 1, y: s.probs[cls] }));
  const series = [];
  for (let c = 0; c < 10; c++) {
    if (c === r.label || c === r.pred) continue;
    series.push({ label: null, points: seriesFor(c), color: GRAY, width: 1 });
  }
  if (r.pred !== r.label) {
    series.push({ label: `predicción ${r.pred}`, points: seriesFor(r.pred), color: "#d64545", width: 2.2 });
  }
  series.push({ label: `etiqueta ${r.label}`, points: seriesFor(r.label), color: "#2f6fed", width: 2.2 });
  drawChart($("chart-trace"), series, { yMin: 0, yMax: 1 });
  $("trace-legend").textContent =
    "Cada paso del deslizamiento en el eje x; probabilidad de cada dígito en el eje y. " +
    "Azul: etiqueta real" + (r.pred !== r.label ? "; rojo: predicción final (errónea)" : "") +
    "; gris: las demás clases.";
}

function startAnim() {
  stopAnim();
  if (!SAMPLE?.trace) return;
  animIdx = 0;
  renderSampleStep();
  $("anim-play").textContent = "⏸ Pausar";
  animTimer = setInterval(() => {
    animIdx += 1;
    if (animIdx >= SAMPLE.trace.steps.length) { stopAnim(); animIdx = SAMPLE.trace.steps.length - 1; }
    renderSampleStep();
  }, 450);
}

function stopAnim() {
  if (animTimer) clearInterval(animTimer);
  animTimer = null;
  $("anim-play").textContent = "▶ Reproducir";
}

$("anim-play").addEventListener("click", () => (animTimer ? stopAnim() : startAnim()));
$("anim-prev").addEventListener("click", () => {
  stopAnim();
  if (SAMPLE?.trace) { animIdx = Math.max(0, animIdx - 1); renderSampleStep(); }
});
$("anim-next").addEventListener("click", () => {
  stopAnim();
  if (SAMPLE?.trace) { animIdx = Math.min(SAMPLE.trace.steps.length - 1, animIdx + 1); renderSampleStep(); }
});

// ============================================================
// PESTAÑA DATASETS
// ============================================================

// ---------- visualizador de deslizamiento (window_size / stride) ----------

let SLIDE_DS = [];   // catálogo completo de datasets (para el visualizador)
let SLIDE = null;    // respuesta de /slide + imagen de fondo cargada
let slideIdx = 0, slideTimer = null;

const loadPng = (b64) => new Promise((res) => {
  const img = new Image();
  img.onload = () => res(img);
  img.src = "data:image/png;base64," + b64;
});

async function initSlidePanel() {
  SLIDE_DS = await api("/api/datasets");
  const sel = $("slide-dataset");
  const prev = sel.value;
  sel.innerHTML = SLIDE_DS.map((d) =>
    `<option value="${d.name}">${d.name}${d.custom ? " (custom)" : ""}</option>`).join("");
  if (prev && SLIDE_DS.some((d) => d.name === prev)) { sel.value = prev; return; }
  onSlideDatasetChange();
}

function onSlideDatasetChange() {
  const ds = SLIDE_DS.find((d) => d.name === $("slide-dataset").value);
  if (!ds) return;
  $("slide-ws").value = ds.defaults.window_size ?? 28;
  $("slide-stride").value = ds.defaults.stride ?? 7;
  refreshSlide();
}

async function refreshSlide() {
  stopSlide();
  const name = $("slide-dataset").value;
  const info = $("slide-info");
  if (!name) return;
  try {
    const q = new URLSearchParams();
    if ($("slide-ws").value) q.set("window_size", $("slide-ws").value);
    if ($("slide-stride").value) q.set("stride", $("slide-stride").value);
    const r = await api(`/api/datasets/${name}/slide?` + q);
    let imgEl = null;
    if (r.full_image) {
      // muestra real del dataset como fondo (mapa de píxeles encima)
      const idx = Math.max(0, parseInt($("slide-index").value, 10) || 0);
      const sq = new URLSearchParams({ split: $("slide-split").value, offset: idx,
                                       limit: 1, params: "{}" });
      const s = await api(`/api/datasets/${name}/samples?` + sq);
      if (s.items.length) imgEl = await loadPng(s.items[0].png);
    }
    SLIDE = { ...r, imgEl };
    slideIdx = 0;
    const perAxis = r.axis_coords.length;
    info.innerHTML =
      `<b>${r.steps} paso${r.steps === 1 ? "" : "s"}</b> — ventana ` +
      `${r.window_size}×${r.window_size}, stride ${r.stride}` +
      `<br>posiciones por eje (${perAxis}): [${r.axis_coords.join(", ")}]` +
      `<br>stride propio del dataset: ${r.dataset_uses_stride
        ? (r.effective_defaults.stride ?? "—") : "no tiene (ventanas aleatorias)"}` +
      (r.note ? `<br><br>${r.note}` : "");
    renderSlideStep();
    startSlide();
  } catch (e) {
    SLIDE = null;
    info.textContent = "Error: " + e.message;
    $("slide-step").textContent = "";
  }
}

function renderSlideStep() {
  if (!SLIDE) return;
  const canvas = $("slide-canvas");
  const ctx = canvas.getContext("2d");
  const n = SLIDE.image_size;
  const scale = canvas.width / n;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (SLIDE.imgEl) ctx.drawImage(SLIDE.imgEl, 0, 0, canvas.width, canvas.height);
  else { ctx.fillStyle = "#f3f5f9"; ctx.fillRect(0, 0, canvas.width, canvas.height); }

  // mapa de píxeles (grilla 28×28)
  ctx.strokeStyle = "rgba(120,130,150,.35)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= n; i++) {
    ctx.beginPath(); ctx.moveTo(i * scale + .5, 0); ctx.lineTo(i * scale + .5, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * scale + .5); ctx.lineTo(canvas.width, i * scale + .5); ctx.stroke();
  }

  const ws = SLIDE.window_size;
  // rastro: posiciones ya visitadas
  ctx.fillStyle = "rgba(47,111,237,.12)";
  for (let j = 0; j < slideIdx; j++) {
    const [t, l] = SLIDE.positions[j];
    ctx.fillRect(l * scale, t * scale, ws * scale, ws * scale);
  }
  // ventana actual
  const [top, left] = SLIDE.positions[slideIdx];
  ctx.strokeStyle = "#2f6fed";
  ctx.lineWidth = 3;
  ctx.strokeRect(left * scale, top * scale, ws * scale, ws * scale);

  $("slide-step").textContent =
    `paso ${slideIdx + 1}/${SLIDE.steps} · esquina (fila, col) = (${top}, ${left})`;
}

function startSlide() {
  stopSlide();
  if (!SLIDE || SLIDE.steps < 2) { if (SLIDE) renderSlideStep(); return; }
  slideIdx = 0;
  renderSlideStep();
  $("slide-play").textContent = "⏸ Pausar";
  slideTimer = setInterval(() => {
    slideIdx += 1;
    if (slideIdx >= SLIDE.steps) { stopSlide(); slideIdx = SLIDE.steps - 1; }
    renderSlideStep();
  }, 350);
}

function stopSlide() {
  if (slideTimer) clearInterval(slideTimer);
  slideTimer = null;
  $("slide-play").textContent = "▶ Reproducir";
}

$("slide-dataset").addEventListener("change", onSlideDatasetChange);
["slide-ws", "slide-stride", "slide-split", "slide-index"].forEach((id) =>
  $(id).addEventListener("change", refreshSlide));
$("slide-play").addEventListener("click", () => (slideTimer ? stopSlide() : startSlide()));
$("slide-prev").addEventListener("click", () => {
  stopSlide();
  if (SLIDE) { slideIdx = Math.max(0, slideIdx - 1); renderSlideStep(); }
});
$("slide-next").addEventListener("click", () => {
  stopSlide();
  if (SLIDE) { slideIdx = Math.min(SLIDE.steps - 1, slideIdx + 1); renderSlideStep(); }
});

async function refreshDatasetsTab() {
  await initSlidePanel();
  const all = await api("/api/datasets");
  const builtin = all.filter((d) => !d.custom);
  const tbodyB = $("builtin-ds-table").querySelector("tbody");
  tbodyB.innerHTML = builtin.map((d) => `
    <tr><td>${d.name}</td><td class="hint">${d.description}</td>
    <td>${d.compatible_with.join(", ")}</td></tr>`).join("");

  const customs = await api("/api/custom-datasets");
  const tbody = $("custom-ds-table").querySelector("tbody");
  tbody.innerHTML = "";
  customs.forEach((d) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${d.name}</td>
      <td class="hint">${d.base.name}</td>
      <td class="hint">${d.base.split}</td>
      <td>${d.count}</td>
      <td class="hint">${d.description || "—"}</td>
      <td class="actions">
        <button class="small secondary" data-rename="${d.name}" title="Renombrar">✎</button>
        <button class="small secondary" data-desc="${d.name}" title="Editar descripción">🖉</button>
        <button class="small secondary" data-copy="${d.name}" title="Duplicar">⧉</button>
        <button class="small danger" data-del="${d.name}" title="Eliminar">🗑</button>
      </td>`;
    tbody.appendChild(tr);
  });
  const msg = $("custom-ds-msg");
  msg.textContent = customs.length ? "" : "Todavía no hay datasets custom.";
  const handle = (sel, fn) => tbody.querySelectorAll(sel).forEach((b) =>
    b.addEventListener("click", async () => {
      try { await fn(b); await refreshDatasetsTab(); }
      catch (e) { msg.textContent = "Error: " + e.message; }
    }));
  handle("[data-rename]", async (b) => {
    const name = prompt("Nuevo nombre (minúsculas, dígitos, _ o -):", b.dataset.rename);
    if (name === null || name === b.dataset.rename) return;
    await patch(`/api/custom-datasets/${b.dataset.rename}`, { new_name: name });
  });
  handle("[data-desc]", async (b) => {
    const d = prompt("Descripción:");
    if (d === null) return;
    await patch(`/api/custom-datasets/${b.dataset.desc}`, { description: d });
  });
  handle("[data-copy]", async (b) => {
    await post(`/api/custom-datasets/${b.dataset.copy}/copy`);
    msg.textContent = "Duplicado.";
  });
  handle("[data-del]", async (b) => {
    if (!confirm(`¿Eliminar el dataset custom ${b.dataset.del}?`)) return;
    await del(`/api/custom-datasets/${b.dataset.del}`);
    msg.textContent = "Eliminado.";
  });
}

// ============================================================
// init
// ============================================================

function fillDigitSelects() {
  const opts = `<option value="">todas</option>` +
    [...Array(10).keys()].map((d) => `<option value="${d}">${d}</option>`).join("");
  $("filter-label").innerHTML = opts;
  $("filter-pred").innerHTML = opts;
}

$("nn-select").addEventListener("change", onNNChange);
$("dataset-select").addEventListener("change", onDatasetChange);
$("init-from-select").addEventListener("change", onInitFromChange);
$("btn-train").addEventListener("click", startTraining);

$("test-exp-select").addEventListener("change", onTestExpChange);
$("test-dataset-select").addEventListener("change", onTestDatasetChange);
$("btn-evaluate").addEventListener("click", startEvaluation);
$("btn-browse").addEventListener("click", browseSamples);
$("btn-save-ds").addEventListener("click", saveFilteredDataset);
["filter-result", "filter-ambiguity", "filter-label", "filter-pred"].forEach((id) =>
  $(id).addEventListener("change", () => { PAGE.offset = 0; loadResultsPage(); }));
$("grid-prev").addEventListener("click", () => {
  if (PAGE.offset > 0) { PAGE.offset = Math.max(0, PAGE.offset - PAGE.limit); loadResultsPage(); }
});
$("grid-next").addEventListener("click", () => {
  if (PAGE.offset + PAGE.limit < PAGE.total) { PAGE.offset += PAGE.limit; loadResultsPage(); }
});

fillDigitSelects();
loadNNs();
refreshExperiments();
setInterval(() => {
  const active = document.querySelector(".tab.active")?.dataset.tab;
  if (active === "entrenar") { refreshExperiments(); refreshDetail(); }
  if (active === "probar") refreshEvals();
}, 2000);
