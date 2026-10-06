// Plyntos Vision AI Master Combined Application Engine
let plyntosChart = null;
let rocChart = null;
let thChart = null;
let impChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPlyntosChart();
  initRocChart();
  initThChart();
  initImpChart();
  onPlyntosSimChange();
});

// Segmented Navigation Tab Switcher
function initTabs() {
  const navPills = document.querySelectorAll(".plyntos-pill");
  const tabContents = document.querySelectorAll(".tab-content");

  navPills.forEach(pill => {
    pill.addEventListener("click", (e) => {
      e.preventDefault();
      const targetTab = pill.getAttribute("data-tab");

      navPills.forEach(p => p.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      pill.classList.add("active");
      const tabEl = document.getElementById(`tab-${targetTab}`);
      if (tabEl) tabEl.classList.add("active");
    });
  });
}

function switchTab(tabId) {
  const pill = document.querySelector(`.plyntos-pill[data-tab="${tabId}"]`);
  if (pill) pill.click();
}

// Light/Dark Theme Toggle
function toggleTheme() {
  const body = document.getElementById("app-body");
  const btnIcon = document.querySelector("#theme-toggle-btn i");

  if (body.classList.contains("dark-theme")) {
    body.classList.remove("dark-theme");
    body.classList.add("light-theme");
    btnIcon.className = "fa-solid fa-moon";
  } else {
    body.classList.remove("light-theme");
    body.classList.add("dark-theme");
    btnIcon.className = "fa-solid fa-sun";
  }
}

// Global Table Search Filter
function filterTable(query) {
  const q = query.toLowerCase().trim();
  const rows = document.querySelectorAll("#orders-tbody tr");

  rows.forEach(row => {
    const text = row.innerText.toLowerCase();
    row.style.display = text.includes(q) ? "" : "none";
  });
}

// Real-Time Customer Predictor Simulator
function onPlyntosSimChange() {
  const tenure = parseInt(document.getElementById("sim-tenure").value);
  const charges = parseFloat(document.getElementById("sim-charges").value);
  const contract = document.getElementById("sim-contract").value;
  const service = document.getElementById("sim-service").value;

  document.getElementById("sim-tenure-val").innerText = `${tenure} mo`;
  document.getElementById("sim-charges-val").innerText = `$${charges.toFixed(2)}`;

  let logit = 0.2;
  logit -= tenure * 0.035;
  logit += (charges - 50.0) * 0.02;

  if (contract === "Month-to-month") logit += 1.35;
  else if (contract === "One year") logit -= 0.55;
  else if (contract === "Two year") logit -= 1.65;

  if (service === "Fiber optic") logit += 0.85;
  else if (service === "No") logit -= 0.75;

  const prob = 1 / (1 + Math.exp(-logit));
  const churnPct = Math.min(Math.max(Math.round(prob * 100), 5), 95);

  // Update Ring
  const ringPct = document.getElementById("ring-percentage");
  if (ringPct) ringPct.innerText = `${churnPct}%`;

  const circle = document.getElementById("ring-progress-circle");
  if (circle) {
    const offset = Math.round(389 - (churnPct / 100) * 389);
    circle.setAttribute("stroke-dashoffset", offset);
  }

  // Update Predictor Tab Prob Gauge
  const predProb = document.getElementById("predictor-prob");
  if (predProb) predProb.innerText = `${churnPct}%`;

  const gaugeRing = document.getElementById("gauge-ring");
  if (gaugeRing) {
    gaugeRing.style.background = `conic-gradient(var(--blue-primary) 0% ${churnPct}%, var(--border-light) ${churnPct}% 100%)`;
  }

  // Update KPIs
  const atRiskRev = Math.round(284650 * (churnPct / 62));
  const ordersCount = Math.round(7043 * (1 + (72 - tenure) / 250));
  const refundRate = (churnPct * 0.42).toFixed(1);

  document.getElementById("kpi-revenue").innerText = `$${atRiskRev.toLocaleString()}`;
  document.getElementById("kpi-orders").innerText = ordersCount.toLocaleString();
  document.getElementById("kpi-churn-rate").innerText = `${refundRate}%`;

  // Update Tooltip
  const ttSales = Math.round(2220 * (1 + (72 - tenure) / 300));
  const ttRev = (142.9 * (churnPct / 62)).toFixed(1);

  const salesEl = document.getElementById("tt-sales");
  if (salesEl) salesEl.innerText = ttSales.toLocaleString();

  const revEl = document.getElementById("tt-rev");
  if (revEl) revEl.innerText = `1,655 ($${ttRev}k)`;

  if (plyntosChart) {
    plyntosChart.data.datasets[0].data[0] = Math.round(75 * (churnPct / 62));
    plyntosChart.data.datasets[1].data[0] = Math.round(35 * (churnPct / 62));
    plyntosChart.update();
  }
}

// Threshold Optimization Simulator
function updateThresholdMetrics() {
  const th = parseFloat(document.getElementById("th-slider").value);
  document.getElementById("th-val").innerText = th.toFixed(2);

  let recall = Math.min(100, Math.round(100 - (th - 0.1) * 90));
  let precision = Math.min(100, Math.round(45 + th * 55));
  let f1 = (2 * (recall / 100) * (precision / 100)) / (recall / 100 + precision / 100);
  let cost = Math.round((100 - recall) * 50 + (100 - precision) * 15 + 150);

  document.getElementById("th-recall").innerText = `${recall.toFixed(1)}%`;
  document.getElementById("th-precision").innerText = `${precision.toFixed(1)}%`;
  document.getElementById("th-f1").innerText = f1.toFixed(4);
  document.getElementById("th-cost").innerText = `$${cost}.00`;
}

// Chart 1: Plyntos 7-Day / Contract Dual Bar
function initPlyntosChart() {
  const ctx = document.getElementById("plyntosPerformanceChart")?.getContext("2d");
  if (!ctx) return;

  plyntosChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Month-to-Month", "One Year", "Two Year", "DSL", "Fiber Optic", "Paperless", "Auto-Pay"],
      datasets: [
        {
          label: "Retained",
          data: [75, 55, 30, 80, 85, 50, 48],
          backgroundColor: [
            "var(--blue-primary)",
            "var(--border-light)",
            "var(--border-light)",
            "var(--border-light)",
            "var(--border-light)",
            "var(--border-light)",
            "var(--border-light)"
          ],
          borderRadius: 8,
          barPercentage: 0.5,
          categoryPercentage: 0.6
        },
        {
          label: "Churned",
          data: [35, 25, 18, 48, 45, 28, 26],
          backgroundColor: [
            "var(--text-sub)",
            "var(--bg-app)",
            "var(--bg-app)",
            "var(--bg-app)",
            "var(--bg-app)",
            "var(--bg-app)",
            "var(--bg-app)"
          ],
          borderRadius: 8,
          barPercentage: 0.5,
          categoryPercentage: 0.6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#64748B", font: { family: "Inter", size: 11, weight: "600" } } },
        y: { min: 0, max: 100, ticks: { stepSize: 25, color: "#94A3B8" }, grid: { color: "#E2E8F0" } }
      }
    }
  });
}

// Chart 2: ROC Curves
function initRocChart() {
  const ctx = document.getElementById("chartRoc")?.getContext("2d");
  if (!ctx) return;

  rocChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
      datasets: [
        {
          label: "Gradient Boosting (AUC = 0.9015)",
          data: [0.0, 0.45, 0.72, 0.84, 0.90, 0.94, 0.96, 0.98, 0.99, 1.0, 1.0],
          borderColor: "var(--blue-primary)",
          backgroundColor: "var(--blue-soft)",
          fill: true,
          tension: 0.3
        },
        {
          label: "Random Forest (AUC = 0.8845)",
          data: [0.0, 0.40, 0.68, 0.80, 0.86, 0.91, 0.94, 0.97, 0.98, 0.99, 1.0],
          borderColor: "var(--text-sub)",
          fill: false,
          tension: 0.3
        },
        {
          label: "Logistic Regression (AUC = 0.8690)",
          data: [0.0, 0.35, 0.62, 0.75, 0.82, 0.87, 0.91, 0.94, 0.97, 0.99, 1.0],
          borderColor: "var(--text-light)",
          fill: false,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#64748B" } } },
      scales: {
        x: { ticks: { color: "#94A3B8" }, grid: { color: "#E2E8F0" } },
        y: { ticks: { color: "#94A3B8" }, grid: { color: "#E2E8F0" } }
      }
    }
  });
}

// Chart 3: Threshold Simulator Curve
function initThChart() {
  const ctx = document.getElementById("chartThreshold")?.getContext("2d");
  if (!ctx) return;

  thChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"],
      datasets: [
        {
          label: "Recall (Caught Churners)",
          data: [100.0, 92.86, 92.86, 85.71, 82.14, 71.43, 57.14, 42.86, 21.43],
          borderColor: "var(--blue-primary)",
          tension: 0.2
        },
        {
          label: "Precision (Flagged Accuracy)",
          data: [41.18, 50.00, 61.90, 75.00, 82.14, 83.33, 88.89, 85.71, 100.00],
          borderColor: "var(--text-sub)",
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#64748B" } } },
      scales: {
        x: { ticks: { color: "#94A3B8" }, grid: { color: "#E2E8F0" } },
        y: { ticks: { color: "#94A3B8" }, grid: { color: "#E2E8F0" } }
      }
    }
  });
}

// Chart 4: Feature Importance
function initImpChart() {
  const ctx = document.getElementById("chartFeatureImportance")?.getContext("2d");
  if (!ctx) return;

  impChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [
        "Contract (Month-to-Month)",
        "Tenure (Months)",
        "Fiber Optic Internet",
        "Monthly Charges ($)",
        "Total Charges ($)",
        "No Tech Support",
        "Paperless Billing",
        "Electronic Check Payment"
      ],
      datasets: [
        {
          label: "Importance (%)",
          data: [32.45, 21.10, 14.80, 12.35, 6.20, 4.15, 3.20, 2.40],
          backgroundColor: "var(--blue-primary)",
          borderRadius: 6
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94A3B8" }, grid: { color: "#E2E8F0" } },
        y: { ticks: { color: "#64748B" }, grid: { color: "#E2E8F0" } }
      }
    }
  });
}
