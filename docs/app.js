const IDX = {
  date: 0, lead_days: 1, actual_tmax: 2, actual_tmin: 3,
  forecast_tmax: 4, forecast_tmin: 5, error_tmax: 6, error_tmin: 7,
  is_hot_anomaly: 8,
};

const COLORS = {
  normal: getCssVar("--series-normal"),
  hot: getCssVar("--series-hot"),
  overall: getCssVar("--series-overall"),
  text: getCssVar("--text-primary"),
  muted: getCssVar("--text-muted"),
  grid: getCssVar("--grid"),
};

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const state = {
  locations: [],
  currentData: null,   // {columns, rows} for the selected location
  allLocationsCache: {},
  charts: {},
};

async function fetchJSON(url, retries = 2) {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
      return await r.json();
    } catch (err) {
      if (attempt >= retries) throw err;
    }
  }
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
function normCDF(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function pValueApprox(z) { return 2 * (1 - normCDF(Math.abs(z))); }

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function variance(arr, m) { return arr.reduce((a, b) => a + (b - m) ** 2, 0) / (arr.length - 1); }

function welchZ(a, b) {
  if (a.length < 2 || b.length < 2) return null;
  const ma = mean(a), mb = mean(b);
  const va = variance(a, ma), vb = variance(b, mb);
  const se = Math.sqrt(va / a.length + vb / b.length);
  if (se === 0) return null;
  return (ma - mb) / se;
}

function getISOWeekYear(dateStr) {
  const d = new Date(dateStr + "T00:00:00Z");
  const dayNum = (d.getUTCDay() + 6) % 7;
  d.setUTCDate(d.getUTCDate() - dayNum + 3);
  const firstThursday = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
  const ftDay = (firstThursday.getUTCDay() + 6) % 7;
  firstThursday.setUTCDate(firstThursday.getUTCDate() - ftDay + 3);
  const week = 1 + Math.round((d - firstThursday) / (7 * 86400000));
  return { isoYear: d.getUTCFullYear(), isoWeek: week };
}

function filterRows(data, from, to) {
  return data.rows.filter(r => r[IDX.date] >= from && r[IDX.date] <= to);
}

function summarizeByLead(rows, metric) {
  const errIdx = metric === "tmax" ? IDX.error_tmax : IDX.error_tmin;
  const byLead = new Map();
  for (const r of rows) {
    const lead = r[IDX.lead_days];
    const err = r[errIdx];
    if (err === null) continue;
    if (!byLead.has(lead)) byLead.set(lead, { all: [], hot: [], normal: [] });
    const bucket = byLead.get(lead);
    bucket.all.push(err);
    (r[IDX.is_hot_anomaly] ? bucket.hot : bucket.normal).push(err);
  }
  const leads = [...byLead.keys()].sort((a, b) => a - b);
  return leads.map(lead => {
    const { all, hot, normal } = byLead.get(lead);
    const z = welchZ(hot, normal);
    return {
      lead_days: lead,
      n: all.length,
      mae: mean(all.map(Math.abs)),
      rmse: Math.sqrt(mean(all.map(v => v * v))),
      bias: mean(all),
      n_hot: hot.length,
      bias_hot: hot.length ? mean(hot) : null,
      n_normal: normal.length,
      bias_normal: normal.length ? mean(normal) : null,
      p_value: z === null ? null : pValueApprox(z),
    };
  });
}

function destroyChart(key) {
  if (state.charts[key]) { state.charts[key].destroy(); delete state.charts[key]; }
}

function baseLineOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: COLORS.text, usePointStyle: true } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}°C` } },
    },
    scales: {
      x: { title: { display: true, text: "Délai (jours)", color: COLORS.text }, ticks: { color: COLORS.muted }, grid: { color: COLORS.grid } },
      y: { title: { display: true, text: yLabel, color: COLORS.text }, ticks: { color: COLORS.muted }, grid: { color: COLORS.grid } },
    },
  };
}

function renderBiasChart(summary) {
  destroyChart("bias");
  const ctx = document.getElementById("bias-chart");
  state.charts.bias = new Chart(ctx, {
    type: "line",
    data: {
      labels: summary.map(s => s.lead_days),
      datasets: [
        { label: "Ensemble", data: summary.map(s => s.bias), borderColor: COLORS.overall, borderDash: [6, 4], pointRadius: 4, tension: 0 },
        { label: "Jours normaux", data: summary.map(s => s.bias_normal), borderColor: COLORS.normal, backgroundColor: COLORS.normal, pointRadius: 5, borderWidth: 2, tension: 0 },
        { label: "Jours de chaleur anomale", data: summary.map(s => s.bias_hot), borderColor: COLORS.hot, backgroundColor: COLORS.hot, pointRadius: 5, borderWidth: 2, tension: 0 },
      ],
    },
    options: baseLineOptions("Erreur moyenne, réel − prévision (°C)"),
  });
}

function renderErrorChart(summary) {
  destroyChart("error");
  const ctx = document.getElementById("error-chart");
  state.charts.error = new Chart(ctx, {
    type: "line",
    data: {
      labels: summary.map(s => s.lead_days),
      datasets: [
        { label: "MAE", data: summary.map(s => s.mae), borderColor: COLORS.normal, backgroundColor: COLORS.normal, pointRadius: 5, borderWidth: 2, tension: 0 },
        { label: "RMSE", data: summary.map(s => s.rmse), borderColor: COLORS.hot, backgroundColor: COLORS.hot, pointRadius: 5, borderWidth: 2, tension: 0 },
      ],
    },
    options: baseLineOptions("Erreur (°C)"),
  });
}

function renderStatTiles(summary) {
  const row = document.getElementById("stat-row");
  const at7 = summary.find(s => s.lead_days === 7) || summary[summary.length - 1];
  if (!at7) { row.innerHTML = ""; return; }
  const gap = (at7.bias_hot ?? NaN) - (at7.bias_normal ?? NaN);
  const pText = at7.p_value === null ? "n/d" : at7.p_value < 0.001 ? "< 0.001" : at7.p_value.toFixed(3);
  const tiles = [
    { label: `Jours dans l'échantillon`, value: at7.n },
    { label: `Jours de chaleur anomale`, value: at7.n_hot },
    { label: `Biais, normal (${at7.lead_days}j)`, value: fmt(at7.bias_normal) },
    { label: `Biais, chaleur (${at7.lead_days}j)`, value: fmt(at7.bias_hot) },
    { label: `Écart`, value: fmt(gap) },
    { label: `p-value (approx.)`, value: pText },
  ];
  row.innerHTML = tiles.map(t => `
    <div class="stat-tile">
      <div class="value">${t.value}</div>
      <div class="label">${t.label}</div>
    </div>`).join("");
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const s = v.toFixed(2);
  return v > 0 ? `+${s}°C` : `${s}°C`;
}

async function renderLocationsComparison(metric, leadDays) {
  destroyChart("locations");
  const names = state.locations.map(l => l.name);
  await Promise.all(names.map(async name => {
    if (!state.allLocationsCache[name]) {
      state.allLocationsCache[name] = await fetchJSON(`data/${name}.json`);
    }
  }));

  const results = state.locations.map(loc => {
    const data = state.allLocationsCache[loc.name];
    const rows = data.rows.filter(r => r[IDX.lead_days] === leadDays);
    const errIdx = metric === "tmax" ? IDX.error_tmax : IDX.error_tmin;
    const hot = rows.filter(r => r[IDX.is_hot_anomaly] && r[errIdx] !== null).map(r => r[errIdx]);
    const normal = rows.filter(r => !r[IDX.is_hot_anomaly] && r[errIdx] !== null).map(r => r[errIdx]);
    return {
      name: loc.name,
      bias_hot: hot.length ? mean(hot) : null,
      bias_normal: normal.length ? mean(normal) : null,
      n_hot: hot.length,
    };
  }).sort((a, b) => (b.bias_hot ?? -999) - (a.bias_hot ?? -999));

  const ctx = document.getElementById("locations-chart");
  state.charts.locations = new Chart(ctx, {
    type: "bar",
    data: {
      labels: results.map(r => `${r.name} (n=${r.n_hot})`),
      datasets: [
        { label: "Jours normaux", data: results.map(r => r.bias_normal), backgroundColor: COLORS.normal },
        { label: "Jours de chaleur anomale", data: results.map(r => r.bias_hot), backgroundColor: COLORS.hot },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: COLORS.text, usePointStyle: true } } },
      scales: {
        x: { ticks: { color: COLORS.muted, maxRotation: 30, minRotation: 30 }, grid: { display: false } },
        y: { title: { display: true, text: "Erreur moyenne (°C)", color: COLORS.text }, ticks: { color: COLORS.muted }, grid: { color: COLORS.grid } },
      },
    },
  });
}

function colorForValue(v, vmax) {
  if (v === null || Number.isNaN(v)) return "transparent";
  const t = Math.max(-1, Math.min(1, v / vmax));
  const neutral = [240, 239, 236];
  const blue = [42, 120, 214];
  const red = [227, 73, 72];
  const target = t < 0 ? blue : red;
  const a = Math.abs(t);
  const rgb = neutral.map((c, i) => Math.round(c + (target[i] - c) * a));
  return `rgb(${rgb.join(",")})`;
}

function renderHeatmap(rows) {
  const lead = 7;
  const byCell = new Map();
  for (const r of rows) {
    if (r[IDX.lead_days] !== lead || r[IDX.error_tmax] === null) continue;
    const { isoYear, isoWeek } = getISOWeekYear(r[IDX.date]);
    const key = `${isoYear}-${isoWeek}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(r[IDX.error_tmax]);
  }
  const years = [...new Set([...byCell.keys()].map(k => +k.split("-")[0]))].sort();
  const weeks = Array.from({ length: 53 }, (_, i) => i + 1);
  let vmax = 0.1;
  for (const vals of byCell.values()) vmax = Math.max(vmax, Math.abs(mean(vals)));

  let html = '<table class="heatmap-table"><thead><tr><th></th>';
  for (const w of weeks) html += `<th>${w % 4 === 0 ? w : ""}</th>`;
  html += "</tr></thead><tbody>";
  for (const y of years) {
    html += `<tr><td class="heatmap-row-label">${y}</td>`;
    for (const w of weeks) {
      const vals = byCell.get(`${y}-${w}`);
      const v = vals ? mean(vals) : null;
      const color = colorForValue(v, vmax);
      const title = v === null ? "" : `${y} semaine ${w} : ${v.toFixed(2)}°C`;
      html += `<td style="background:${color}" title="${title}"></td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("heatmap-wrap").innerHTML = html;
}

function render() {
  const metric = document.getElementById("metric-select").value;
  const from = document.getElementById("date-from").value;
  const to = document.getElementById("date-to").value;
  if (!state.currentData || !from || !to) return;

  const rows = filterRows(state.currentData, from, to);
  const summary = summarizeByLead(rows, metric);
  renderBiasChart(summary);
  renderErrorChart(summary);
  renderStatTiles(summary);
  renderHeatmap(rows);
  renderLocationsComparison(metric, 7);
}

async function loadLocation(name) {
  if (!state.allLocationsCache[name]) {
    state.allLocationsCache[name] = await fetchJSON(`data/${name}.json`);
  }
  state.currentData = state.allLocationsCache[name];
  const dates = state.currentData.rows.map(r => r[IDX.date]);
  const min = dates.reduce((a, b) => (a < b ? a : b));
  const max = dates.reduce((a, b) => (a > b ? a : b));
  const fromEl = document.getElementById("date-from");
  const toEl = document.getElementById("date-to");
  fromEl.min = min; fromEl.max = max;
  toEl.min = min; toEl.max = max;
  if (!fromEl.value) fromEl.value = min;
  if (!toEl.value) toEl.value = max;
  render();
}

async function init() {
  state.locations = await fetchJSON("data/locations.json");
  const sel = document.getElementById("location-select");
  sel.innerHTML = state.locations.map(l =>
    `<option value="${l.name}">${l.name[0].toUpperCase() + l.name.slice(1)} — ${l.region}</option>`
  ).join("");

  sel.addEventListener("change", () => loadLocation(sel.value));
  document.getElementById("metric-select").addEventListener("change", render);
  document.getElementById("date-from").addEventListener("change", render);
  document.getElementById("date-to").addEventListener("change", render);

  await loadLocation(state.locations[0].name);
}

init();
