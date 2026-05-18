const statusEl = document.getElementById("status");
const gridEl = document.getElementById("grid");
const refreshBtn = document.getElementById("refreshBtn");
const retrainBtn = document.getElementById("retrainBtn");
const datasetNameInput = document.getElementById("datasetNameInput");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function clearGrid() {
  while (gridEl.firstChild) {
    gridEl.removeChild(gridEl.firstChild);
  }
}

function createCard(dp) {
  const card = document.createElement("div");
  card.className = "card";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = dp.filename || `datapoint ${dp.id}`;
  img.src = `/api/datapoints/${dp.id}/image`;
  img.addEventListener("click", () => {
    window.location.href = `/annotate/${dp.id}`;
  });

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `id=${dp.id} dataset=${dp.dataset_name || "?"} file=${dp.filename || "?"}`;

  card.appendChild(img);
  card.appendChild(meta);

  const annotateLink = document.createElement("a");
  annotateLink.className = "button";
  annotateLink.href = `/annotate/${dp.id}`;
  annotateLink.textContent = "Annotate";

  card.appendChild(annotateLink);
  return card;
}

async function loadUnlabeled() {
  setStatus("Loading unlabeled images…");
  clearGrid();

  const response = await fetch("/api/unlabeled?limit=50");
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || "Request failed");
  }

  const datapoints = Array.isArray(data.datapoints) ? data.datapoints : [];
  setStatus(`Found ${datapoints.length} unlabeled images.`);

  for (const dp of datapoints) {
    gridEl.appendChild(createCard(dp));
  }

  // If dataset input is empty, prefill with the first datapoint's dataset_name
  if (datapoints.length && datasetNameInput && !datasetNameInput.value) {
    datasetNameInput.value = datapoints[0].dataset_name || "";
  }
}

refreshBtn.addEventListener("click", () => {
  loadUnlabeled().catch((err) => setStatus(`Error: ${err.message}`, true));
});

loadUnlabeled().catch((err) => setStatus(`Error: ${err.message}`, true));

if (retrainBtn) {
  retrainBtn.addEventListener("click", async () => {
    const datasetName = datasetNameInput ? datasetNameInput.value.trim() : "";
    if (!confirm(`Trigger retrain on dataset '${datasetName || 'yolo_pose_demo'}'?`)) {
      return;
    }

    retrainBtn.disabled = true;
    setStatus("Triggering retrain…");

    try {
      const runName = `${datasetName || 'all'}_${new Date().toISOString().replace(/[:.]/g, "-")}`;
      const response = await fetch("/api/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_name: datasetName, experiment_name: "yolo_pose_demo", run_name: runName }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.error || JSON.stringify(data));
      }
      setStatus("Retrain triggered.");
      console.info("Retrain response:", data);
    } catch (err) {
      setStatus(`Retrain failed: ${err.message}`, true);
    } finally {
      retrainBtn.disabled = false;
    }
  });
}