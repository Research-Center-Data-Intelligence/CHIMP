const webcam = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const startInferBtn = document.getElementById("startInferBtn");
const stopInferBtn = document.getElementById("stopInferBtn");
const statusEl = document.getElementById("status");

const drawCtx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 640;
captureCanvas.height = 640;
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

let activeStream = null;
let inferTimer = null;
let inferInFlight = false;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser does not support webcam access.", true);
    return;
  }

  try {
    activeStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    webcam.srcObject = activeStream;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startInferBtn.disabled = false;
    setStatus("Camera is live.");
  } catch (error) {
    setStatus(`Unable to start camera: ${error.message}`, true);
  }
}

function stopCamera() {
  stopInferenceLoop();

  if (!activeStream) {
    return;
  }

  activeStream.getTracks().forEach((track) => track.stop());
  webcam.srcObject = null;
  activeStream = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  startInferBtn.disabled = true;
  stopInferBtn.disabled = true;
  clearOverlay();
  setStatus("Camera stopped.");
}

function clearOverlay() {
  drawCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawDetections(detections) {
  clearOverlay();

  drawCtx.lineWidth = 2;
  drawCtx.font = "14px Segoe UI";

  for (const det of detections) {
    const bbox = det.bbox_xyxy || [0, 0, 0, 0];
    const [x1, y1, x2, y2] = bbox;

    drawCtx.strokeStyle = "#00ff88";
    drawCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const conf = det.confidence || 0;
    const label = `${(conf * 100).toFixed(1)}%`;
    drawCtx.fillStyle = "rgba(0, 0, 0, 0.55)";
    drawCtx.fillRect(x1, Math.max(0, y1 - 20), 58, 18);
    drawCtx.fillStyle = "#ffffff";
    drawCtx.fillText(label, x1 + 4, Math.max(13, y1 - 6));

    for (const kp of det.keypoints || []) {
      if ((kp.confidence || 0) < 0.2) {
        continue;
      }
      drawCtx.fillStyle = "#1f6feb";
      drawCtx.beginPath();
      drawCtx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
      drawCtx.fill();
    }
  }
}

async function inferCurrentFrame() {
  if (!activeStream || inferInFlight) {
    return;
  }
  inferInFlight = true;

  try {
    captureCtx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

    const response = await fetch("/api/pose-infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ image: dataUrl })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Inference request failed");
    }

    const detections = Array.isArray(data.detections) ? data.detections : [];
    drawDetections(detections);
    setStatus(`Camera live. Pose detections: ${detections.length}`);
  } catch (error) {
    setStatus(`Pose inference failed: ${error.message}`, true);
  } finally {
    inferInFlight = false;
  }
}

function startInferenceLoop() {
  if (!activeStream || inferTimer) {
    return;
  }

  startInferBtn.disabled = true;
  stopInferBtn.disabled = false;
  setStatus("Camera live. Pose inference running...");

  inferCurrentFrame();
  inferTimer = window.setInterval(inferCurrentFrame, 350);
}

function stopInferenceLoop() {
  if (inferTimer) {
    window.clearInterval(inferTimer);
    inferTimer = null;
  }

  startInferBtn.disabled = !activeStream;
  stopInferBtn.disabled = true;
  if (activeStream) {
    setStatus("Camera is live. Pose inference stopped.");
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
startInferBtn.addEventListener("click", startInferenceLoop);
stopInferBtn.addEventListener("click", stopInferenceLoop);
window.addEventListener("beforeunload", stopCamera);
