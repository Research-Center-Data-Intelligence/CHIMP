import base64
from io import BytesIO

import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def sample_evenly_spaced_indices(total_frames: int, frame_count: int) -> list[int]:
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


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2.0)
    xyxy[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2.0)
    xyxy[:, 2] = boxes[:, 0] + (boxes[:, 2] / 2.0)
    xyxy[:, 3] = boxes[:, 1] + (boxes[:, 3] / 2.0)
    return xyxy


def nms_indices(
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


def decode_yolo_pose_output(
    raw_output, conf_threshold: float = 0.25, iou_threshold: float = 0.45
):
    out = np.asarray(raw_output, dtype=np.float32)
    if out.ndim != 3 or out.shape[0] < 1:
        return []

    pred = np.transpose(out[0], (1, 0)) if out.shape[1] <= out.shape[2] else out[0]
    if pred.ndim != 2 or pred.shape[1] < 6:
        return []

    scores = pred[:, 4]
    mask = scores >= conf_threshold
    pred = pred[mask]
    scores = scores[mask]
    if len(pred) == 0:
        return []

    boxes_xywh = pred[:, :4]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms_indices(boxes_xyxy, scores, iou_threshold=iou_threshold)

    detections = []
    for i in keep:
        row = pred[i]
        keypoint_values = row[5:]
        keypoints = []
        if len(keypoint_values) >= 3 and len(keypoint_values) % 3 == 0:
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


def prepare_tensor_from_data_url(image_data_url: str, size: int = IMAGE_SIZE):
    if not image_data_url.startswith("data:image"):
        raise ValueError("Expected image data URL payload")

    encoded = image_data_url.split(",", maxsplit=1)[-1]
    raw_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)

    # Match notebook preprocessing: NCHW float32 in [0, 1]
    x = arr.transpose(2, 0, 1)[None] / 255.0
    return x


def prepare_tensor_from_image_bytes(image_bytes: bytes, size: int = IMAGE_SIZE) -> tuple[np.ndarray, int, int]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_width, image_height = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)

    # Match the notebook preprocessing: NCHW float32 in [0, 1]
    x = arr.transpose(2, 0, 1)[None] / 255.0
    return x, image_width, image_height