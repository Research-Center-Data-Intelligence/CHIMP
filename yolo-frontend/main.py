import base64
import os
from io import BytesIO

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image


SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://localhost:5254")
MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolo_pose_demo")
MODEL_STAGE = os.environ.get("YOLO_MODEL_STAGE", "production")
MODEL_SESSION_ID = os.environ.get("YOLO_MODEL_SESSION_ID", "")
IMAGE_SIZE = 640
REQUEST_TIMEOUT_SECONDS = 30


app = Flask(__name__)


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


def _infer_pose_tensor(model_input: np.ndarray):
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
    return _decode_yolo_pose_output(first_output)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5260, debug=True)
