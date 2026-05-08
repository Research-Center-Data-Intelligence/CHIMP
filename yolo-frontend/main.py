import base64
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np
import cv2
import requests
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image


SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://localhost:5254")
MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolo_pose_demo")
MODEL_STAGE = os.environ.get("YOLO_MODEL_STAGE", "production")
MODEL_SESSION_ID = os.environ.get("YOLO_MODEL_SESSION_ID", "")
IMAGE_SIZE = 640
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_FRAME_COUNT = 10
CVAT_BASE_URL = os.environ.get("CVAT_BASE_URL", "http://localhost:8088").rstrip("/")
CVAT_HOST_HEADER = os.environ.get("CVAT_HOST_HEADER", "").strip()
CVAT_API_TOKEN = os.environ.get("CVAT_API_TOKEN", "")
CVAT_PROJECT_ID = os.environ.get("CVAT_PROJECT_ID", "").strip()
CVAT_TASK_NAME_PREFIX = os.environ.get("CVAT_TASK_NAME_PREFIX", "yolo-webcam").strip() or "yolo-webcam"
CVAT_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("CVAT_REQUEST_TIMEOUT_SECONDS", "120"))


app = Flask(__name__)


@dataclass
class CVATContext:
    base_url: str
    session: requests.Session


def _sample_evenly_spaced_indices(total_frames: int, frame_count: int) -> list[int]:
    if total_frames <= 0 or frame_count <= 0:
        return []
    if total_frames <= frame_count:
        return list(range(total_frames))

    raw_indices = np.linspace(0, total_frames - 1, num=frame_count)
    indices = []
    seen = set()
    for value in raw_indices:
        index = int(round(float(value)))
        index = max(0, min(total_frames - 1, index))
        if index not in seen:
            seen.add(index)
            indices.append(index)

    if len(indices) < frame_count:
        for index in range(total_frames):
            if index not in seen:
                seen.add(index)
                indices.append(index)
                if len(indices) == frame_count:
                    break

    indices.sort()
    return indices[:frame_count]


def _extract_png_frames_from_video(video_bytes: bytes, frame_count: int = DEFAULT_FRAME_COUNT) -> list[tuple[str, bytes]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero")

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as temp_file:
        temp_file.write(video_bytes)
        temp_file.flush()

        capture = cv2.VideoCapture(temp_file.name)
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0:
                frame_indices = _sample_evenly_spaced_indices(total_frames, frame_count)

                if not frame_indices:
                    raise ValueError("Uploaded video did not contain any readable frames")

                extracted_frames: list[tuple[str, bytes]] = []
                current_index = 0
                target_positions = {index: position for position, index in enumerate(frame_indices)}

                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    if current_index in target_positions:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb_frame)
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        frame_name = f"frame_{target_positions[current_index]:02d}_{current_index:06d}.png"
                        extracted_frames.append((frame_name, buffer.getvalue()))

                        if len(extracted_frames) == len(frame_indices):
                            break

                    current_index += 1

                if not extracted_frames:
                    raise ValueError("No frames could be extracted from the uploaded video")

                return extracted_frames

            all_frames: list[np.ndarray] = []
            while True:
                success, frame = capture.read()
                if not success:
                    break
                all_frames.append(frame)

            if not all_frames:
                raise ValueError("Uploaded video did not contain any readable frames")

            frame_indices = _sample_evenly_spaced_indices(len(all_frames), frame_count)
            extracted_frames = []
            for output_position, frame_index in enumerate(frame_indices):
                rgb_frame = cv2.cvtColor(all_frames[frame_index], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                frame_name = f"frame_{output_position:02d}_{frame_index:06d}.png"
                extracted_frames.append((frame_name, buffer.getvalue()))

            return extracted_frames
        finally:
            capture.release()


def _cvat_parse_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload)
    except Exception:  # noqa: BLE001
        return response.text[:600] or "Unknown CVAT error"


def _cvat_request(
    session: requests.Session,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> requests.Response:
    merged_headers = dict(headers or {})
    if CVAT_HOST_HEADER:
        merged_headers.setdefault("Host", CVAT_HOST_HEADER)
    return session.request(
        method=method.upper(),
        url=f"{CVAT_BASE_URL}{path}",
        headers=merged_headers,
        timeout=CVAT_REQUEST_TIMEOUT_SECONDS,
        **kwargs,
    )


def _create_cvat_task(session: requests.Session, task_name: str, project_id: int | None = None) -> int:
    payload: dict[str, Any] = {
        "name": task_name,
        "labels": [
            {"name": "person", "color": "#FF0000", "attributes": []}
        ],
    }
    if project_id is not None:
        payload["project_id"] = int(project_id)

    response = _cvat_request(session, "POST", "/api/tasks", json=payload)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Failed to create CVAT task (status={response.status_code}): {_cvat_parse_error(response)}"
        )

    data = response.json()
    task_id = data.get("id")
    if task_id is None:
        raise RuntimeError("CVAT task creation response did not include task id")
    return int(task_id)




def _upload_frames_to_cvat_task(
    session: requests.Session,
    task_id: int,
    frames: list[tuple[str, bytes]],
) -> str | None:
    endpoint = f"/api/tasks/{task_id}/data/"

    start_response = _cvat_request(session, "POST", endpoint, headers={"Upload-Start": "1"})
    if start_response.status_code not in (200, 202):
        raise RuntimeError(
            f"CVAT upload start failed (status={start_response.status_code}): {_cvat_parse_error(start_response)}"
        )

    opened_buffers: list[BytesIO] = []
    files = []
    try:
        for index, (frame_name, frame_bytes) in enumerate(frames):
            frame_buffer = BytesIO(frame_bytes)
            opened_buffers.append(frame_buffer)
            files.append((f"client_files[{index}]", (frame_name, frame_buffer, "image/png")))

        multiple_response = _cvat_request(
            session,
            "POST",
            endpoint,
            headers={"Upload-Multiple": "1"},
            data={"image_quality": "100"},
            files=files,
        )
        if multiple_response.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"CVAT upload failed (status={multiple_response.status_code}): {_cvat_parse_error(multiple_response)}"
            )
    finally:
        for frame_buffer in opened_buffers:
            frame_buffer.close()

    finish_response = _cvat_request(
        session,
        "POST",
        endpoint,
        headers={"Upload-Finish": "1"},
        json={"image_quality": 100, "sorting_method": "lexicographical"},
    )
    if finish_response.status_code >= 400:
        raise RuntimeError(
            f"CVAT upload finalize failed (status={finish_response.status_code}): {_cvat_parse_error(finish_response)}"
        )

    payload = finish_response.json() if finish_response.content else {}
    return payload.get("rq_id")


def _send_frames_to_cvat(
    extracted_frames: list[tuple[str, bytes]],
    *,
    task_name: str,
    project_id: int | None,
) -> dict[str, Any]:
    if not CVAT_API_TOKEN:
        raise RuntimeError("CVAT_API_TOKEN is not configured")

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {CVAT_API_TOKEN}"})

    user_response = _cvat_request(session, "GET", "/api/users/self")
    if user_response.status_code >= 400:
        raise RuntimeError(
            f"CVAT authentication failed (status={user_response.status_code}): {_cvat_parse_error(user_response)}"
        )

    task_id = _create_cvat_task(session, task_name=task_name, project_id=project_id)
    rq_id = _upload_frames_to_cvat_task(session, task_id=task_id, frames=extracted_frames)

    return {
        "task_id": task_id,
        "task_url": f"{CVAT_BASE_URL}/tasks/{task_id}",
        "rq_id": rq_id,
        "frame_count": len(extracted_frames),
    }


def wait_for_request(ctx: CVATContext, request_id: str, poll_seconds: int = 5) -> dict[str, Any]:
    while True:
        response = _cvat_request(ctx.session, "GET", f"/api/requests/{request_id}")
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to poll CVAT request {request_id} (status={response.status_code}): {_cvat_parse_error(response)}"
            )

        payload = response.json() if response.content else {}
        status = str(payload.get("status", "")).lower()
        if status == "finished":
            return payload
        if status in {"failed", "canceled", "cancelled"}:
            raise RuntimeError(f"CVAT request {request_id} did not complete: {payload}")

        time.sleep(poll_seconds)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2.0)
    xyxy[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2.0)
    xyxy[:, 2] = boxes[:, 0] + (boxes[:, 2] / 2.0)
    xyxy[:, 3] = boxes[:, 1] + (boxes[:, 3] / 2.0)
    return xyxy


def _nms_indices(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int32)


def _decode_yolo_pose_output(
    raw_output, conf_threshold: float = 0.25, iou_threshold: float = 0.45
):
    out = np.asarray(raw_output, dtype=np.float32)
    if out.ndim != 3 or out.shape[0] < 1:
        return []

    pred = np.transpose(out[0], (1, 0)) if out.shape[1] <= out.shape[2] else out[0]
    if pred.shape[1] < 6:
        return []

    scores = pred[:, 4]
    mask = scores >= conf_threshold
    pred = pred[mask]
    scores = scores[mask]
    if len(pred) == 0:
        return []

    boxes_xywh = pred[:, :4]
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    keep = _nms_indices(boxes_xyxy, scores, iou_threshold=iou_threshold)

    detections = []
    for i in keep:
        row = pred[i]
        keypoint_values = row[5:]
        keypoints = []
        if len(keypoint_values) >= 3:
            kp = keypoint_values.reshape(-1, 3)
            keypoints = [
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "confidence": float(point[2]),
                }
                for point in kp
            ]

        detections.append(
            {
                "confidence": float(row[4]),
                "bbox_xywh": [float(v) for v in row[:4]],
                "bbox_xyxy": [float(v) for v in boxes_xyxy[i]],
                "keypoints": keypoints,
            }
        )

    return detections


def _prepare_tensor_from_data_url(image_data_url: str):
    if not image_data_url.startswith("data:image"):
        raise ValueError("Expected image data URL payload")

    encoded = image_data_url.split(",", maxsplit=1
    )[-1]
    raw_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)

    # Match notebook preprocessing: NCHW float32 in [0, 1]
    x = arr.transpose(2, 0, 1)[None] / 255.0
    return x


def _infer_pose_tensor(
    model_input: np.ndarray,
    conf_threshold: float = 0.10,
    iou_threshold: float = 0.45,
):
    infer_url = f"{SERVING_API_URL}/model/{MODEL_NAME}/infer"
    params = {"stage": MODEL_STAGE}
    if MODEL_SESSION_ID:
        params["id"] = MODEL_SESSION_ID

    payload = {"inputs": model_input.tolist()}
    response = requests.post(
        infer_url,
        params=params,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    result = response.json()
    predictions = result.get("predictions")
    if not isinstance(predictions, dict) or not predictions:
        return []

    raw_outputs = predictions.get("raw", predictions)
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        return []

    first_output = np.asarray(next(iter(raw_outputs.values())), dtype=np.float32)
    return _decode_yolo_pose_output(first_output, conf_threshold=conf_threshold, iou_threshold=iou_threshold)


def _prepare_tensor_from_image_bytes(image_bytes: bytes, size: int = IMAGE_SIZE) -> tuple[np.ndarray, int, int]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_width, image_height = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)

    # Match the notebook preprocessing: NCHW float32 in [0, 1]
    x = arr.transpose(2, 0, 1)[None] / 255.0
    return x, image_width, image_height


def _coco_keypoints_from_detection(
    keypoints: list[dict],
    scale_x: float,
    scale_y: float,
    kp_conf_threshold: float,
) -> tuple[list[float], int]:
    coco_keypoints: list[float] = []
    num_keypoints = 0

    for index in range(17):
        if index < len(keypoints):
            keypoint = keypoints[index]
            x = float(keypoint.get("x", 0.0)) * scale_x
            y = float(keypoint.get("y", 0.0)) * scale_y
            confidence = float(keypoint.get("confidence", 0.0))
            visibility = 2 if confidence >= kp_conf_threshold else 1
            if confidence >= kp_conf_threshold:
                num_keypoints += 1
            coco_keypoints.extend([x, y, visibility])
        else:
            coco_keypoints.extend([0.0, 0.0, 0])

    return coco_keypoints, num_keypoints


def _build_coco_keypoints_predictions_from_frames(
    extracted_frames: list[tuple[str, bytes]],
    *,
    model_input_size: int = IMAGE_SIZE,
    det_conf_threshold: float = 0.10,
    kp_conf_threshold: float = 0.20,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for frame_name, frame_bytes in extracted_frames:
        image_tensor, image_width, image_height = _prepare_tensor_from_image_bytes(
            frame_bytes,
            size=model_input_size,
        )
        scale_x = float(image_width) / float(model_input_size)
        scale_y = float(image_height) / float(model_input_size)

        images.append(
            {
                "id": next_image_id,
                "file_name": frame_name,
                "width": image_width,
                "height": image_height,
            }
        )

        detections = _infer_pose_tensor(image_tensor, conf_threshold=det_conf_threshold)
        for detection in detections:
            bbox_xywh = detection.get("bbox_xywh", [0.0, 0.0, 0.0, 0.0])
            if len(bbox_xywh) != 4:
                continue

            center_x, center_y, width, height = bbox_xywh
            x = (float(center_x) - float(width) / 2.0) * scale_x
            y = (float(center_y) - float(height) / 2.0) * scale_y
            width_scaled = float(width) * scale_x
            height_scaled = float(height) * scale_y

            coco_keypoints, num_keypoints = _coco_keypoints_from_detection(
                detection.get("keypoints", []),
                scale_x=scale_x,
                scale_y=scale_y,
                kp_conf_threshold=kp_conf_threshold,
            )

            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [x, y, width_scaled, height_scaled],
                    "area": max(0.0, width_scaled) * max(0.0, height_scaled),
                    "num_keypoints": num_keypoints,
                    "keypoints": coco_keypoints,
                    "score": float(detection.get("confidence", 0.0)),
                }
            )
            next_annotation_id += 1

        next_image_id += 1

    return {
        "info": {"description": "CHIMP generated pre-annotations", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
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
                ],
                "skeleton": [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                    [2, 4], [3, 5], [4, 6], [5, 7],
                ],
            }
        ],
    }


def _import_coco_predictions_to_cvat_task(
    session: requests.Session,
    task_id: int,
    coco_predictions: dict[str, Any],
    format_name: str = "COCO Keypoints 1.0",
) -> str | None:
    annotation_bytes = json.dumps(coco_predictions).encode("utf-8")
    annotation_file = BytesIO(annotation_bytes)

    response = _cvat_request(
        session,
        "POST",
        f"/api/tasks/{task_id}/annotations/",
        params={"format": format_name},
        files={"annotation_file": ("prelabels_coco_keypoints.json", annotation_file, "application/json")},
    )

    if response.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"Failed to import pre-annotations (status={response.status_code}): {_cvat_parse_error(response)}"
        )

    payload = response.json() if response.content else {}
    return payload.get("rq_id")


def _download_annotations_as_coco(
    session: requests.Session,
    task_id: int,
    format_name: str = "COCO Keypoints 1.0",
) -> dict | None:
    """Attempt to download task annotations in COCO Keypoints format.

    Returns parsed JSON on success, None otherwise.
    """
    try:
        resp = _cvat_request(
            session,
            "GET",
            f"/api/tasks/{task_id}/annotations/",
            params={"format": format_name, "action": "download"},
        )
        if resp.status_code >= 400:
            print(f"[_download_annotations_as_coco] download failed: {resp.status_code} {_cvat_parse_error(resp)}")
            return None

        # CVAT may return JSON directly or bytes representing JSON
        try:
            return resp.json()
        except Exception:
            try:
                return json.loads(resp.content.decode("utf-8"))
            except Exception as ex:  # noqa: BLE001
                print(f"[_download_annotations_as_coco] failed to parse content: {ex}")
                return None
    except Exception as ex:  # noqa: BLE001
        print(f"[_download_annotations_as_coco] exception: {ex}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/pose-infer", methods=["POST"])
def pose_infer():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    body = request.get_json(silent=True) or {}
    image_data_url = body.get("image")
    if not isinstance(image_data_url, str) or not image_data_url:
        return jsonify({"error": "Missing image data URL in 'image' field"}), 400

    try:
        tensor = _prepare_tensor_from_data_url(image_data_url)
        detections = _infer_pose_tensor(tensor)
        return jsonify({"detections": detections})
    except requests.HTTPError as ex:
        return (
            jsonify(
                {
                    "error": "Serving API returned an error",
                    "details": str(ex),
                }
            ),
            502,
        )
    except requests.RequestException as ex:
        return (
            jsonify(
                {
                    "error": "Could not reach serving API",
                    "details": str(ex),
                }
            ),
            502,
        )
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Inference failed", "details": str(ex)}), 500


@app.route("/api/video-to-png-frames", methods=["POST"])
def video_to_png_frames():
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "Missing video file in 'video' field"}), 400

    frame_count = request.form.get("frame_count", default=DEFAULT_FRAME_COUNT, type=int)
    if frame_count is None or frame_count <= 0:
        return jsonify({"error": "frame_count must be a positive integer"}), 400

    try:
        video_bytes = video_file.read()
        extracted_frames = _extract_png_frames_from_video(video_bytes, frame_count=frame_count)

        output_zip = BytesIO()
        with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for frame_name, frame_bytes in extracted_frames:
                archive.writestr(frame_name, frame_bytes)

        output_zip.seek(0)
        return send_file(
            output_zip,
            mimetype="application/zip",
            as_attachment=True,
            download_name="video_frames.zip",
        )
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Video frame extraction failed", "details": str(ex)}), 500


@app.route("/api/video-to-cvat-task", methods=["POST"])
def video_to_cvat_task():
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "Missing video file in 'video' field"}), 400

    frame_count = request.form.get("frame_count", default=DEFAULT_FRAME_COUNT, type=int)
    if frame_count is None or frame_count <= 0:
        return jsonify({"error": "frame_count must be a positive integer"}), 400

    requested_project_id = request.form.get("project_id", default=None, type=int)
    configured_project_id = int(CVAT_PROJECT_ID) if CVAT_PROJECT_ID.isdigit() else None
    project_id = requested_project_id if requested_project_id is not None else configured_project_id

    requested_task_name = (request.form.get("task_name") or "").strip()
    auto_task_name = f"{CVAT_TASK_NAME_PREFIX}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    task_name = requested_task_name or auto_task_name

    try:
        if not CVAT_API_TOKEN:
            return jsonify({"error": "CVAT_API_TOKEN is not configured"}), 500

        video_bytes = video_file.read()
        extracted_frames = _extract_png_frames_from_video(video_bytes, frame_count=frame_count)

        session = requests.Session()
        session.headers.update({"Authorization": f"Token {CVAT_API_TOKEN}"})
        ctx = CVATContext(base_url=CVAT_BASE_URL, session=session)

        user_response = _cvat_request(session, "GET", "/api/users/self")
        if user_response.status_code >= 400:
            return jsonify({"error": f"CVAT authentication failed (status={user_response.status_code}): {_cvat_parse_error(user_response)}"}), 502

        task_id = _create_cvat_task(session, task_name=task_name, project_id=project_id)
        upload_rq_id = _upload_frames_to_cvat_task(session, task_id=task_id, frames=extracted_frames)
        if upload_rq_id:
            wait_for_request(ctx, upload_rq_id, poll_seconds=2)

        coco_predictions = _build_coco_keypoints_predictions_from_frames(
            extracted_frames,
            model_input_size=IMAGE_SIZE,
            det_conf_threshold=0.10,
            kp_conf_threshold=0.20,
        )
        annotation_count = len(coco_predictions.get("annotations", []))
        import_rq_id = None
        prelabels_imported = False
        annotations_in_task = 0
        if annotation_count > 0:
            try:
                import_rq_id = _import_coco_predictions_to_cvat_task(session, task_id, coco_predictions)
                if import_rq_id:
                    import_request_payload = wait_for_request(ctx, import_rq_id, poll_seconds=2)
                    print(f"[video_to_cvat_task] import request payload: {import_request_payload}")
                    prelabels_imported = True
                    # verify imported annotations by attempting to download them
                    downloaded = _download_annotations_as_coco(session, task_id)
                    if isinstance(downloaded, dict):
                        found_annotations = downloaded.get("annotations") or downloaded.get("annotations", [])
                        try:
                            annotations_in_task = len(found_annotations) if isinstance(found_annotations, list) else 0
                        except Exception:
                            annotations_in_task = 0
                    else:
                        annotations_in_task = 0
            except RuntimeError as ex:
                print(f"[video_to_cvat_task] Pre-annotation import failed: {ex}")
                prelabels_imported = False
                annotations_in_task = 0
                import_request_payload = None

        return jsonify(
            {
                "message": "Frames and pre-annotations uploaded to CVAT task",
                "task_id": task_id,
                "task_url": f"{CVAT_BASE_URL}/tasks/{task_id}",
                "upload_rq_id": upload_rq_id,
                "import_rq_id": import_rq_id,
                "import_request_payload": import_request_payload,
                "frame_count": len(extracted_frames),
                "annotation_count": annotation_count,
                "prelabels_imported": prelabels_imported,
                "annotations_in_task": annotations_in_task,
                "task_name": task_name,
                "project_id": project_id,
            }
        )
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except RuntimeError as ex:
        return jsonify({"error": str(ex)}), 502
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to create CVAT task", "details": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5260, debug=True)
