/* SIH26006 – FreightAI Dashboard JavaScript */
"use strict";

// ══════════════════ State ══════════════════
let selectedModel = "xgboost";
let gaugeChart = null;
let trendChart = null;
let featChart  = null;
let compareChart = null;

// ══════════════════ Model Toggle ══════════════════
function setModel(model) {
  selectedModel = model;
  document.getElementById("btn-xgb").classList.toggle("active", model === "xgboost");
  document.getElementById("btn-lr").classList.toggle("active",  model === "linear");
  document.getElementById("result-model-tag").textContent =
    model === "xgboost" ? "XGBoost Prediction" : "Linear Regression Prediction";
}

// ══════════════════ Predict ══════════════════
async function runPrediction() {
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>&nbsp; Predicting…`;

  const payload = {
    route:               document.getElementById("route").value,
    vessel_type:         document.getElementById("vessel_type").value,
    cargo_type:          document.getElementById("cargo_type").value,
    cargo_volume_mt:     parseFloat(document.getElementById("cargo_volume").value),
    bunker_fuel_usd_mt:  parseFloat(document.getElementById("bunker_fuel").value),
    month:               parseInt(document.getElementById("month").value),
    year:                2025,
    port_congestion_idx: parseFloat(document.getElementById("congestion").value),
    model:               selectedModel,
  };

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const json = await res.json();
    if (json.status === "success") {
      displayResults(json.data);
    } else {
      alert("Prediction error: " + json.message);
    }
  } catch (err) {
    alert("Network error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">🚀</span> Predict Freight Rate`;
  }
}

// ══════════════════ Display Results ══════════════════
function displayResults(data) {
  const rate    = data.predicted_rate_usd_t;
  const altRate = data.alt_model_rate_usd_t;
  const p       = data.procurement;

  // Rate card
  animateNumber("rate-value", rate, 2);
  document.getElementById("rate-inr").textContent =
    `≈ ₹ ${(rate * 83.5).toFixed(2)} per tonne`;
  const altLabel = selectedModel === "xgboost" ? "Linear Regression" : "XGBoost";
  document.getElementById("rate-alt").textContent =
    `${altLabel}: $${altRate.toFixed(2)}/tonne`;

  // Rate bands (±10%, ±20%)
  document.getElementById("band-low-val").textContent  = `$${(rate * 0.90).toFixed(2)}`;
  document.getElementById("band-mid-val").textContent  = `$${rate.toFixed(2)}`;
  document.getElementById("band-high-val").textContent = `$${(rate * 1.10).toFixed(2)}`;
  document.getElementById("rate-bands").style.display = "flex";

  // Procurement
  document.getElementById("proc-vessel").textContent  = p.recommended_vessel;
  document.getElementById("proc-count").textContent   = p.vessels_required;
  document.getElementById("proc-cost-usd").textContent= `$${p.total_freight_usd.toLocaleString()}`;
  document.getElementById("proc-cost-inr").textContent= `₹${p.total_freight_inr.toLocaleString()}`;
  document.getElementById("proc-util").textContent    = `${p.vessel_utilisation}%`;
  document.getElementById("proc-daily").textContent   = `$${p.daily_rate_usd.toLocaleString()}/day`;

  // Decision banner
  const banner = document.getElementById("decision-banner");
  banner.textContent = p.decision;
  banner.className = "decision-banner";
  if (p.decision.includes("🟢")) banner.classList.add("green");
  else if (p.decision.includes("🟡")) banner.classList.add("yellow");
  else if (p.decision.includes("🟠")) banner.classList.add("orange");
  else if (p.decision.includes("🔴")) banner.classList.add("red");

  // Gauge chart
  updateGauge(rate);
}

// ══════════════════ Animate Number ══════════════════
function animateNumber(id, target, decimals = 2) {
  const el = document.getElementById(id);
  const start = parseFloat(el.textContent) || 0;
  const duration = 800;
  const startTime = performance.now();

  function step(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    const current  = start + (target - start) * ease;
    el.textContent = current.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ══════════════════ Gauge Chart ══════════════════
function updateGauge(rate) {
  const max = 50;
  const val = Math.min(rate, max);
  const remaining = max - val;

  // Color by rate
  let color = "#56d364";
  if (rate >= 28)     color = "#ff7b72";
  else if (rate >= 18) color = "#f0883e";
  else if (rate >= 10) color = "#e3b341";

  if (gaugeChart) gaugeChart.destroy();
  const ctx = document.getElementById("gaugeChart").getContext("2d");
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [val, remaining],
        backgroundColor: [color, "rgba(48,54,61,0.3)"],
        borderWidth: 0,
        circumference: 240,
        rotation: 240,
      }],
    },
    options: {
      cutout: "72%",
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
      animation: { animateRotate: true, duration: 800 },
    },
    plugins: [{
      id: "gaugeLabel",
      afterDraw(chart) {
        const { ctx, chartArea: { left, top, right, bottom } } = chart;
        const cx = (left + right) / 2;
        const cy = (top + bottom) / 2 + 20;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = color;
        ctx.font = "bold 22px 'JetBrains Mono', monospace";
        ctx.fillText(`$${rate.toFixed(1)}`, cx, cy - 10);
        ctx.fillStyle = "#8b949e";
        ctx.font = "11px Inter";
        ctx.fillText("USD/tonne", cx, cy + 14);
        ctx.restore();
      },
    }],
  });
}

// ══════════════════ Historical Trend Chart ══════════════════
async function loadTrendChart() {
  const data = await fetch("/api/historical").then(r => r.json());
  if (!data || data.length === 0) return;

  const labels = data.map(d => d.month);
  const avg    = data.map(d => d.avg_rate);
  const minR   = data.map(d => d.min_rate);
  const maxR   = data.map(d => d.max_rate);

  const ctx = document.getElementById("trendChart").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Avg Rate ($/t)",
          data: avg,
          borderColor: "#4fc3f7",
          backgroundColor: "rgba(79,195,247,0.08)",
          borderWidth: 2.5,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
        },
        {
          label: "Max Rate",
          data: maxR,
          borderColor: "rgba(240,136,62,0.5)",
          borderDash: [4, 4],
          borderWidth: 1.5,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        },
        {
          label: "Min Rate",
          data: minR,
          borderColor: "rgba(86,211,100,0.5)",
          borderDash: [4, 4],
          borderWidth: 1.5,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        },
      ],
    },
    options: darkChartOptions("Freight Rate ($/tonne)"),
  });
}

// ══════════════════ Feature Importance Chart ══════════════════
async function loadFeatureChart() {
  const data = await fetch("/api/feature_importance").then(r => r.json());
  if (!data.labels || data.labels.length === 0) return;

  const ctx = document.getElementById("featChart").getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 400, 0);
  gradient.addColorStop(0, "rgba(79,195,247,0.9)");
  gradient.addColorStop(1, "rgba(86,211,100,0.9)");

  featChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Importance",
        data: data.values,
        backgroundColor: gradient,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      ...darkChartOptions("Feature Importance"),
      indexAxis: "y",
    },
  });
}

// ══════════════════ Model Comparison Chart ══════════════════
async function loadCompareChart() {
  const data = await fetch("/api/metrics").then(r => r.json());
  if (!data || data.length === 0) return;

  // Update hero stats
  const xgb = data.find(m => m.model === "XGBoost");
  if (xgb) {
    document.getElementById("stat-rmse").textContent = xgb.RMSE.toFixed(2);
    document.getElementById("stat-r2").textContent   = xgb.R2.toFixed(3);
  }

  // Build metrics rows
  const metricsRow = document.getElementById("metrics-row");
  const metricConfigs = [
    { key: "RMSE",   label: "RMSE ($/t)",   color: "blue",   lower: true },
    { key: "MAE",    label: "MAE ($/t)",    color: "green",  lower: true },
    { key: "R2",     label: "R² Score",     color: "purple", lower: false },
    { key: "MAPE_%", label: "MAPE (%)",     color: "orange", lower: true },
  ];

  metricsRow.innerHTML = "";
  for (const cfg of metricConfigs) {
    const lr  = data.find(m => m.model === "Linear Regression");
    const xgb = data.find(m => m.model === "XGBoost");
    if (!lr || !xgb) continue;

    const lrVal  = lr[cfg.key];
    const xgbVal = xgb[cfg.key];
    const better = cfg.lower ? (xgbVal < lrVal ? "XGB" : "LR") : (xgbVal > lrVal ? "XGB" : "LR");

    metricsRow.innerHTML += `
      <div class="metric-box">
        <div class="metric-name">${cfg.label}</div>
        <div style="display:flex;gap:12px;margin-top:8px;">
          <div>
            <div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:3px;">LINEAR</div>
            <div class="metric-val ${cfg.color}">${lrVal}</div>
          </div>
          <div>
            <div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:3px;">XGBOOST</div>
            <div class="metric-val ${cfg.color}">${xgbVal}</div>
          </div>
        </div>
        <div style="font-size:0.68rem;color:var(--accent-green);margin-top:8px;">
          ✓ ${better} wins
        </div>
      </div>
    `;
  }

  // Compare bar chart
  const labels  = data.map(m => m.model);
  const r2vals  = data.map(m => m.R2);
  const ctx = document.getElementById("compareChart").getContext("2d");
  compareChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "R² Score",
          data: r2vals,
          backgroundColor: ["rgba(79,195,247,0.7)", "rgba(86,211,100,0.7)"],
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      ...darkChartOptions("R² Score"),
      scales: {
        ...darkChartOptions().scales,
        y: { ...darkChartOptions().scales?.y, min: 0, max: 1 },
      },
    },
  });
}

// ══════════════════ Shared Chart Options ══════════════════
function darkChartOptions(yLabel = "") {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 1000 },
    plugins: {
      legend: {
        labels: { color: "#8b949e", font: { family: "Inter", size: 11 }, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: "rgba(22,27,34,0.95)",
        titleColor: "#e6edf3",
        bodyColor: "#8b949e",
        borderColor: "rgba(48,54,61,0.8)",
        borderWidth: 1,
        padding: 10,
      },
    },
    scales: {
      x: {
        ticks: { color: "#8b949e", font: { size: 10 }, maxTicksLimit: 12 },
        grid:  { color: "rgba(48,54,61,0.4)" },
      },
      y: {
        title: { display: !!yLabel, text: yLabel, color: "#8b949e", font: { size: 10 } },
        ticks: { color: "#8b949e", font: { size: 10 } },
        grid:  { color: "rgba(48,54,61,0.4)" },
      },
    },
  };
}

// ══════════════════ Batch CSV Upload ══════════════════
async function handleCSVUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const outputEl = document.getElementById("batch-output");
  outputEl.innerHTML = `<div class="empty-state"><span class="spinner"></span>&nbsp; Processing ${file.name}…</div>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch("/batch_predict", { method: "POST", body: formData });
    const json = await res.json();
    if (json.status === "success") {
      displayBatchResults(json.results, json.count);
    } else {
      outputEl.innerHTML = `<div class="empty-state" style="color:var(--accent-red)">Error: ${json.message}</div>`;
    }
  } catch (err) {
    outputEl.innerHTML = `<div class="empty-state" style="color:var(--accent-red)">Network error: ${err.message}</div>`;
  }
}

function displayBatchResults(results, count) {
  const outputEl = document.getElementById("batch-output");
  if (!results.length) {
    outputEl.innerHTML = `<div class="empty-state">No results returned</div>`;
    return;
  }

  let html = `<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px;">
                ${count} shipments processed
              </div>
              <table class="batch-table">
                <thead>
                  <tr>
                    <th>#</th><th>Route</th><th>Rate ($/t)</th><th>Decision</th>
                  </tr>
                </thead><tbody>`;

  results.forEach((r, i) => {
    if (r.error) {
      html += `<tr><td>${i+1}</td><td colspan="3" style="color:var(--accent-red)">${r.error}</td></tr>`;
    } else {
      const decColor = r.decision.includes("🟢") ? "var(--accent-green)"
                     : r.decision.includes("🟡") ? "#e3b341"
                     : r.decision.includes("🟠") ? "var(--accent-orange)"
                     : "var(--accent-red)";
      html += `<tr>
        <td style="color:var(--text-muted)">${i+1}</td>
        <td>${r.route}</td>
        <td style="font-family:'JetBrains Mono',monospace">$${r.predicted_rate}</td>
        <td style="color:${decColor};font-size:0.78rem">${r.decision}</td>
      </tr>`;
    }
  });

  html += `</tbody></table>`;
  outputEl.innerHTML = html;
}

// ══════════════════ Download Sample CSV ══════════════════
function downloadSampleCSV() {
  const rows = [
    "route,vessel_type,cargo_type,cargo_volume_mt,bunker_fuel_usd_mt,month,year",
    "Paradip→Visakhapatnam,Supramax,Iron Ore,42000,620,11,2025",
    "Kandla→Mundra,Handysize,Coal,22000,580,3,2025",
    "Mumbai→Haldia,Panamax,Steel Coils,65000,700,7,2025",
    "Haldia→Ennore,Capesize,Limestone,120000,650,1,2025",
    "Ennore→Mumbai,Supramax,Pig Iron,50000,710,10,2025",
  ];
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "sample_freight_batch.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════ Smooth Nav ══════════════════
document.querySelectorAll(".nav-pill").forEach(pill => {
  pill.addEventListener("click", e => {
    document.querySelectorAll(".nav-pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
  });
});

// ══════════════════ Init ══════════════════
document.addEventListener("DOMContentLoaded", () => {
  loadTrendChart();
  loadFeatureChart();
  loadCompareChart();

  // Init gauge with placeholder
  updateGauge(0);
});
