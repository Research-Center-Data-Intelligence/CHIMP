const COCO_KEYPOINTS = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
];

const datapointId = window.__DATAPOINT_ID__;
const statusEl = document.getElementById("status");
const detailsEl = document.getElementById("details");
const imageEl = document.getElementById("annotateImage");
const overlayEl = document.getElementById("overlay");
const keypointsEl = document.getElementById("keypoints");
const progressEl = document.getElementById("progress");
const currentPointEl = document.getElementById("currentPoint");
const instructionEl = document.getElementById("instruction");
const refreshBtn = document.getElementById("refreshBtn");
const undoBtn = document.getElementById("undoBtn");
const clearBtn = document.getElementById("clearBtn");
const saveBtn = document.getElementById("saveBtn");

const state = {
  details: null,
  points: Array.from({ length: 17 }, () => null),
  activeIndex: 0,
  naturalWidth: 0,
  naturalHeight: 0,
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function pointsComplete() {
  return state.points.every((point) => point && Number.isFinite(point.x) && Number.isFinite(point.y));
}

function hasAnyLabelled() {
  return state.points.some((point) => Boolean(point));
}

function nextMissingIndex(startIndex = 0) {
  for (let index = startIndex; index < state.points.length; index += 1) {
    if (!state.points[index]) {
      return index;
    }
  }
  return state.points.findIndex((point) => point === null);
}

function computeBbox(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const padding = Math.max(4, Math.round(Math.max(width, height) * 0.05));

  return [
    Math.max(0, Math.round(minX - padding)),
    Math.max(0, Math.round(minY - padding)),
    Math.round(width + padding * 2),
    Math.round(height + padding * 2),
  ];
}

function buildAnnotation() {
  // Build keypoints array of length 51. Points not labeled become [0,0,0].
  const pointsArr = state.points.map((p) => p || null);
  const keypoints = [];
  for (const p of pointsArr) {
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      keypoints.push(0, 0, 0);
    } else {
      keypoints.push(Math.round(p.x), Math.round(p.y), 2);
    }
  }

  const num_keypoints = keypoints.reduce((acc, val, idx) => (idx % 3 === 2 && val > 0 ? acc + 1 : acc), 0);

  // compute bbox from visible/labeled points (v>0)
  const labeledPoints = state.points.filter((p) => p && p.v && Number.isFinite(p.x) && Number.isFinite(p.y));
  const bbox = labeledPoints.length ? computeBbox(labeledPoints) : [0, 0, state.naturalWidth || 1, state.naturalHeight || 1];

  return {
    format: "coco_keypoints",
    category_id: 1,
    bbox,
    keypoints,
    num_keypoints,
    image_width: state.naturalWidth,
    image_height: state.naturalHeight,
    area: bbox[2] * bbox[3],
  };
}

function renderPointList() {
  clearNode(keypointsEl);

  COCO_KEYPOINTS.forEach((name, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "kp";
    if (index === state.activeIndex) {
      button.classList.add("active");
    }
    const p = state.points[index];
    if (p) {
      button.classList.add("done");
    }

    const labelText = p ? ` (${Math.round(p.x)}, ${Math.round(p.y)})` : "";
    button.textContent = `${index + 1}. ${name}${labelText}`;
    button.addEventListener("click", () => {
      state.activeIndex = index;
      render();
    });
    keypointsEl.appendChild(button);
  });
}

function renderMarkers() {
  clearNode(overlayEl);

  if (!state.naturalWidth || !state.naturalHeight) {
    return;
  }

  state.points.forEach((point, index) => {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return;
    }

    const marker = document.createElement("div");
    marker.className = "marker";
    marker.textContent = String(index + 1);
    marker.style.left = `${(point.x / state.naturalWidth) * 100}%`;
    marker.style.top = `${(point.y / state.naturalHeight) * 100}%`;
    overlayEl.appendChild(marker);
  });
}

function render() {
  const placedCount = state.points.filter(Boolean).length;
  progressEl.textContent = `${placedCount}/17 points placed`;
  currentPointEl.textContent = `Current point: ${COCO_KEYPOINTS[state.activeIndex]}`;
  instructionEl.textContent = "Click the image to place the selected keypoint. Use the list to jump to any joint. Toggle 'not visible' for occluded joints.";
  saveBtn.disabled = !hasAnyLabelled();

  renderPointList();
  renderMarkers();
}

function resetAll() {
  state.points = Array.from({ length: 17 }, () => null);
  state.activeIndex = 0;
  render();
}

function undoLast() {
  for (let index = state.points.length - 1; index >= 0; index -= 1) {
    if (state.points[index]) {
      state.points[index] = null;
      state.activeIndex = index;
      break;
    }
  }
  render();
}

function imageCoordsFromEvent(event) {
  const rect = imageEl.getBoundingClientRect();
  const x = clamp((event.clientX - rect.left) * (state.naturalWidth / rect.width), 0, state.naturalWidth);
  const y = clamp((event.clientY - rect.top) * (state.naturalHeight / rect.height), 0, state.naturalHeight);
  return { x, y };
}

async function loadDetails() {
  const response = await fetch(`/api/datapoints/${datapointId}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || "Could not load datapoint details");
  }

  state.details = data;
  detailsEl.textContent = `id=${data.id} dataset=${data.dataset_name || "?"} file=${data.filename || "?"}`;
  imageEl.src = `/api/datapoints/${datapointId}/image`;
}

async function saveAnnotation() {
  const annotation = buildAnnotation();
  if (!annotation || !annotation.num_keypoints) {
    setStatus("Mark at least one keypoint as labeled before saving.", true);
    return;
  }

  saveBtn.disabled = true;
  setStatus("Saving annotation…");

  try {
    const response = await fetch(`/api/datapoints/${datapointId}/label`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ annotation }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.error || "Could not save annotation");
    }

    setStatus("Saved annotation.");
    window.location.href = "/";
  } catch (error) {
    setStatus(`Error: ${error.message}`, true);
    saveBtn.disabled = false;
  }
}

imageEl.addEventListener("load", () => {
  state.naturalWidth = imageEl.naturalWidth;
  state.naturalHeight = imageEl.naturalHeight;
  if (!state.activeIndex && state.points[0] === null) {
    state.activeIndex = 0;
  }
  render();
});

overlayEl.addEventListener("click", (event) => {
  if (!state.naturalWidth || !state.naturalHeight) {
    return;
  }

  const { x, y } = imageCoordsFromEvent(event);
  state.points[state.activeIndex] = { x, y };
    const nextIndex = nextMissingIndex(state.activeIndex + 1);
    if (nextIndex !== -1) {
      state.activeIndex = nextIndex;
    }
  render();
});

undoBtn.addEventListener("click", undoLast);
clearBtn.addEventListener("click", resetAll);
saveBtn.addEventListener("click", saveAnnotation);
refreshBtn.addEventListener("click", () => {
  imageEl.src = `/api/datapoints/${datapointId}/image?ts=${Date.now()}`;
});

loadDetails()
  .then(() => {
    setStatus("Ready.");
  })
  .catch((error) => {
    setStatus(`Error: ${error.message}`, true);
  });
