let currentMode = "classification";
let classificationChart = null;
let regressionChart = null;
let lastClassificationData = null;

function setMode(mode) {
  currentMode = mode;
  document
    .querySelectorAll(".mode-btn")
    .forEach((btn) => btn.classList.remove("active"));
  if (mode === "classification") {
    document.getElementById("class-mode").classList.add("active");
  } else {
    document.getElementById("reg-mode").classList.add("active");
  }
}

const API_URL = window.API_URL || 'http://localhost:5000';

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  document.getElementById("upload-screen").classList.add("hidden");
  document.getElementById("loading").classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", currentMode);

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("Oops! Something went wrong: " + data.error);
      location.reload();
      return;
    }

    showResults(data);
  } catch (err) {
    alert("Connection Error! Is the server running?");
    location.reload();
  }
}

function showResults(data) {
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("result-screen").classList.remove("hidden");

  if (data.mode === 'classification') {
    lastClassificationData = data;
    document.getElementById("classification-results").classList.remove("hidden");
    document.getElementById("classification-leaderboard").classList.remove("hidden");
    document.getElementById("model-selector-container").classList.remove("hidden");
    document.getElementById("classification-chart-container").classList.remove("hidden");
    document.getElementById("confusion-matrix-container").classList.remove("hidden");
    document.getElementById("regression-results").classList.add("hidden");

    const tbody = document.getElementById("class-body");
    tbody.innerHTML = "";
    
    const selector = document.getElementById("model-select");
    selector.innerHTML = "";

    for (const [model, info] of Object.entries(data.results)) {
      // Populate Table
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${model}</td>
        <td>${(info.metrics.accuracy * 100).toFixed(2)}%</td>
        <td>${(info.metrics.precision * 100).toFixed(2)}%</td>
        <td>${(info.metrics.recall * 100).toFixed(2)}%</td>
        <td>${(info.metrics.f1 * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(row);

      // Populate Dropdown
      const option = document.createElement("option");
      option.value = model;
      option.innerText = model;
      selector.appendChild(option);
    }

    // Default selection to Random Forest if available
    if (data.results["Random Forest"]) {
      selector.value = "Random Forest";
    }

    updateSelectedModel();
    renderClassificationChart(data.results);
  } else {
    document.getElementById("classification-results").classList.add("hidden");
    document.getElementById("classification-leaderboard").classList.add("hidden");
    document.getElementById("model-selector-container").classList.add("hidden");
    document.getElementById("classification-chart-container").classList.add("hidden");
    document.getElementById("confusion-matrix-container").classList.add("hidden");
    document.getElementById("regression-results").classList.remove("hidden");

    const tbody = document.getElementById("reg-body");
    tbody.innerHTML = "";
    
    for (const [model, metrics] of Object.entries(data.results)) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${model}</td>
        <td>${metrics.MAE}</td>
        <td>${metrics.MSE}</td>
        <td>${metrics.RMSE}</td>
        <td>${metrics.SSE}</td>
      `;
      tbody.appendChild(row);
    }

    renderRegressionChart(data.results);
  }
}

function updateSelectedModel() {
  const modelName = document.getElementById("model-select").value;
  if (!lastClassificationData || !lastClassificationData.results[modelName]) return;

  const info = lastClassificationData.results[modelName];
  
  // Update Confusion Matrix
  document.getElementById("cm-00").innerText = info.cm[0][0];
  document.getElementById("cm-01").innerText = info.cm[0][1];
  document.getElementById("cm-10").innerText = info.cm[1][0];
  document.getElementById("cm-11").innerText = info.cm[1][1];
  
  // Update Title
  document.querySelector("#confusion-matrix-container h3").innerText = `Confusion Matrix (${modelName})`;
}

function renderClassificationChart(results) {
  const ctx = document.getElementById("classificationChart").getContext("2d");
  if (classificationChart) classificationChart.destroy();

  const labels = Object.keys(results);
  const accData = labels.map(l => results[l].metrics.accuracy * 100);

  classificationChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Model Accuracy (%)",
          data: accData,
          backgroundColor: "#4a90e2",
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { 
          beginAtZero: true,
          max: 100
        },
      },
      plugins: {
        legend: { position: "top" },
      }
    },
  });
}

function renderRegressionChart(results) {
  const ctx = document.getElementById("regressionChart").getContext("2d");
  if (regressionChart) regressionChart.destroy();

  const labels = Object.keys(results);
  const maeData = labels.map((l) => results[l].MAE);

  regressionChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Mean Absolute Error (MAE)",
          data: maeData,
          backgroundColor: "#ff8c00",
          borderRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}
