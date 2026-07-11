// Frontend de entrenamiento: selección de NN, datasets compatibles,
// lanzamiento de entrenamientos y visualización en vivo de métricas.

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => {
  if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || r.statusText); });
  return r.json();
});

let NNS = [];
let currentDetail = null;
let pollTimer = null;

// ---------- gráficas (canvas, sin librerías) ----------

const COLORS = ["#2f6fed", "#d64545", "#1a7f37", "#b57d00"];

function drawChart(canvas, series, opts = {}) {
  // series: [{label, points: [{x, y}]}]
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

  series.forEach((s, i) => {
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    s.points.forEach((p, j) => (j ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
    ctx.stroke();
    // leyenda
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fillRect(m.left + 8 + i * 110, m.top, 10, 10);
    ctx.fillStyle = "#1c2330";
    ctx.fillText(s.label, m.left + 22 + i * 110, m.top + 9);
  });
}

// ---------- formulario de entrenamiento ----------

async function loadNNs() {
  NNS = await api("/api/nns");
  const sel = $("nn-select");
  sel.innerHTML = NNS.map((n) => `<option value="${n.name}">${n.name}</option>`).join("");
  await onNNChange();
}

let DATASETS = []; // datasets compatibles con la NN seleccionada (con sus defaults)

async function onNNChange() {
  const nn = $("nn-select").value;
  const spec = NNS.find((n) => n.name === nn);
  $("nn-description").textContent = spec.description;

  DATASETS = await api(`/api/datasets?nn=${nn}`);
  const dsel = $("dataset-select");
  dsel.innerHTML = DATASETS.map((d) => `<option value="${d.name}">${d.name}</option>`).join("");

  const config = JSON.parse(JSON.stringify(spec.defaults));
  const isSeq = nn === "secuenciador";
  $("dim-exp-row").hidden = !isSeq;
  if (isSeq) {
    const dims = await api("/api/experiments?nn=dimensionador&status=completed");
    const dsel2 = $("dim-exp-select");
    dsel2.innerHTML = dims.length
      ? dims.map((e) => `<option value="${e.id}">${e.id} (val_acc=${e.status.best?.val_acc ?? "?"})</option>`).join("")
      : `<option value="">— entrena primero un dimensionador —</option>`;
    config.dimensionador_experiment = dsel2.value || null;
  }
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
      $("config-json").value = JSON.stringify(config, null, 2);
    }
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
    const res = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nn, config }),
    });
    msg.textContent = "Entrenamiento iniciado: " + res.experiment_id;
    await refreshExperiments();
    showDetail(res.experiment_id);
  } catch (e) { msg.textContent = "Error: " + e.message; }
}

// ---------- listado de experimentos ----------

async function refreshExperiments() {
  const exps = await api("/api/experiments");
  const tbody = $("exp-table").querySelector("tbody");
  tbody.innerHTML = "";
  exps.slice().reverse().forEach((e) => {
    const st = e.status.status;
    const tr = document.createElement("tr");
    tr.className = "selectable";
    tr.innerHTML = `
      <td>${e.id}</td>
      <td class="status-${st}">${st}${e.running ? " ⏳" : ""}</td>
      <td>${e.status.best?.val_acc ?? "—"}</td>
      <td>${e.status.final_test?.acc ?? "—"}</td>
      <td>${e.running ? '<button class="small danger" data-stop="' + e.id + '">Detener</button>' : ""}</td>`;
    tr.addEventListener("click", (ev) => {
      if (ev.target.dataset.stop) return;
      showDetail(e.id);
    });
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll("[data-stop]").forEach((b) =>
    b.addEventListener("click", () => api(`/api/experiments/${b.dataset.stop}/stop`, { method: "POST" })));
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

// ---------- init ----------

$("nn-select").addEventListener("change", onNNChange);
$("dataset-select").addEventListener("change", onDatasetChange);
$("btn-train").addEventListener("click", startTraining);

loadNNs();
refreshExperiments();
pollTimer = setInterval(() => { refreshExperiments(); refreshDetail(); }, 2000);
