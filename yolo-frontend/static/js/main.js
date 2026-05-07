const webcam = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const recordVideoBtn = document.getElementById("recordVideoBtn");
const recordingPanel = document.getElementById("recordingPanel");
const recordedVideo = document.getElementById("recordedVideo");
const sendToQueueBtn = document.getElementById("sendToQueueBtn");
const statusEl = document.getElementById("status");

const drawCtx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 640;
captureCanvas.height = 640;
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

let activeStream = null;
let inferTimer = null;
let inferInFlight = false;
let mediaRecorder = null;
let recordingChunks = [];
let recordingPreviewUrl = "";
let recordingActive = false;
let lastRecordingBlob = null;
let lastRecordingMimeType = "video/webm";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setRecordingPanelVisible(isVisible) {
  recordingPanel.classList.toggle("hidden", !isVisible);
}

function setCameraButtonState() {
  const isRunning = Boolean(activeStream);
  recordVideoBtn.disabled = !isRunning;
  recordVideoBtn.textContent = recordingActive ? "Stop Recording" : "Start Recording";
}

function revokeRecordingPreviewUrl() {
  if (recordingPreviewUrl) {
    URL.revokeObjectURL(recordingPreviewUrl);
    recordingPreviewUrl = "";
  }
}

function clearRecordingTimers() {
}

function finalizeRecording(blob, mimeType) {
  console.log("[finalizeRecording] blob:", blob, "size:", blob?.size, "mimeType:", mimeType);
  revokeRecordingPreviewUrl();
  lastRecordingBlob = blob;
  lastRecordingMimeType = mimeType || "video/webm";

  if (!blob || !blob.size) {
    console.log("[finalizeRecording] ERROR: no blob or empty blob");
    setStatus("Recording finished, but no video data was captured.", true);
    setRecordingPanelVisible(false);
    return;
  }

  console.log("[finalizeRecording] Creating object URL and setting up panel");
  recordingPreviewUrl = URL.createObjectURL(blob);
  recordedVideo.src = recordingPreviewUrl;
  console.log("[finalizeRecording] Calling setRecordingPanelVisible(true)");
  setRecordingPanelVisible(true);
  console.log("[finalizeRecording] Recording panel visible property:", !recordingPanel.classList.contains("hidden"));
  setStatus("Recording complete.");
}

function stopRecording() {
  console.log("[stopRecording] Called. recordingActive:", recordingActive, "mediaRecorder state:", mediaRecorder?.state);
  if (!recordingActive && !mediaRecorder) {
    console.log("[stopRecording] No active recording or recorder, returning");
    setCameraButtonState();
    return;
  }

  recordingActive = false;
  clearRecordingTimers();

  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    console.log("[stopRecording] MediaRecorder is null or inactive, returning");
    mediaRecorder = null;
    setCameraButtonState();
    return;
  }

  console.log("[stopRecording] Calling mediaRecorder.stop()");
  setStatus("Finalizing recording...");
  mediaRecorder.stop();
  setCameraButtonState();
}

function startRecording() {
  console.log("[startRecording] Called");
  if (!activeStream) {
    console.log("[startRecording] No active stream");
    setStatus("Start the camera before recording a video.", true);
    return;
  }

  if (!window.MediaRecorder) {
    console.log("[startRecording] No MediaRecorder support");
    setStatus("This browser does not support video recording.", true);
    return;
  }

  try {
    recordingChunks = [];
    setRecordingPanelVisible(false);
    recordingActive = true;

    // Use video-only MIME types since we're not capturing audio
    const preferredMimeTypes = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const selectedMimeType = preferredMimeTypes.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
    console.log("[startRecording] Selected MIME type:", selectedMimeType);
    const recorderOptions = selectedMimeType ? { mimeType: selectedMimeType } : undefined;

    mediaRecorder = recorderOptions ? new MediaRecorder(activeStream, recorderOptions) : new MediaRecorder(activeStream);
    console.log("[startRecording] MediaRecorder created");

    mediaRecorder.ondataavailable = (event) => {
      console.log("[ondataavailable] Chunk received, size:", event.data?.size);
      if (event.data && event.data.size > 0) {
        recordingChunks.push(event.data);
        console.log("[ondataavailable] Total chunks now:", recordingChunks.length);
      }
    };

    mediaRecorder.onstop = () => {
      console.log("[onstop] Recording stopped. Total chunks:", recordingChunks.length);
      const recorderMimeType = mediaRecorder?.mimeType || selectedMimeType || "video/webm";
      const blob = new Blob(recordingChunks, { type: recorderMimeType });
      console.log("[onstop] Blob created. Size:", blob.size, "Type:", recorderMimeType);
      mediaRecorder = null;
      recordingActive = false;
      finalizeRecording(blob, recorderMimeType);
      setCameraButtonState();
    };

    mediaRecorder.start();
    console.log("[startRecording] mediaRecorder.start() called");
    setCameraButtonState();
    setStatus("Recording video... click Stop Recording when finished.");
  } catch (error) {
    mediaRecorder = null;
    recordingActive = false;
    clearRecordingTimers();
    setCameraButtonState();
    setStatus(`Unable to start recording: ${error.message}`, true);
  }
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

    try {
      await webcam.play();
    } catch {
      // Some browsers may auto-play without an explicit play() call.
    }

    setCameraButtonState();
    setStatus("Camera is live. Starting pose inference...");
    startInferenceLoop();
  } catch (error) {
    setStatus(`Unable to start camera: ${error.message}`, true);
  }
}

function stopCamera() {
  stopInferenceLoop();

  if (mediaRecorder) {
    stopRecording();
  }

  if (!activeStream) {
    return;
  }

  activeStream.getTracks().forEach((track) => track.stop());
  webcam.srcObject = null;
  activeStream = null;
  setCameraButtonState();
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

  if (webcam.readyState < 2) {
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

    if (!mediaRecorder) {
      setStatus(`Camera live. Pose detections: ${detections.length}`);
    }
  } catch (error) {
    if (!mediaRecorder) {
      setStatus(`Pose inference failed: ${error.message}`, true);
    }
  } finally {
    inferInFlight = false;
  }
}

function startInferenceLoop() {
  if (!activeStream || inferTimer) {
    return;
  }
  setStatus("Camera live. Pose inference running...");

  inferCurrentFrame();
  inferTimer = window.setInterval(inferCurrentFrame, 350);
}

function stopInferenceLoop() {
  if (inferTimer) {
    window.clearInterval(inferTimer);
    inferTimer = null;
  }

  if (activeStream) {
    setStatus("Camera is live. Pose inference stopped.");
  }
}

// Auto-start camera on page load
window.addEventListener("load", () => {
  startCamera();
});

recordVideoBtn.addEventListener("click", () => {
  if (recordingActive) {
    stopRecording();
  } else {
    startRecording();
  }
});

sendToQueueBtn.addEventListener("click", () => {
  if (!lastRecordingBlob || !lastRecordingBlob.size) {
    setStatus("Record a video before sending it for frame extraction.", true);
    return;
  }

  sendToQueueBtn.disabled = true;
  sendToQueueBtn.textContent = "Extracting Frames...";
  setStatus("Uploading recording for frame extraction...");

  const formData = new FormData();
  const videoFileName = `recording.${lastRecordingMimeType.includes("mp4") ? "mp4" : "webm"}`;
  formData.append("video", lastRecordingBlob, videoFileName);
  formData.append("frame_count", "10");

  fetch("/api/video-to-png-frames", {
    method: "POST",
    body: formData
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "Frame extraction failed");
      }

      return response.blob();
    })
    .then((zipBlob) => {
      console.log("[sendToQueueBtn] Received extracted frames zip:", zipBlob.size, "bytes");
      setStatus("Extracted 10 frames successfully.");
      sendToQueueBtn.textContent = "Frames Extracted";
      sendToQueueBtn.disabled = true;
    })
    .catch((error) => {
      console.error("[sendToQueueBtn] Frame extraction failed:", error);
      setStatus(`Frame extraction failed: ${error.message}`, true);
      sendToQueueBtn.disabled = false;
      sendToQueueBtn.textContent = "Send to Labeling Queue";
    });
});
window.addEventListener("beforeunload", stopCamera);

setCameraButtonState();
setRecordingPanelVisible(false);
