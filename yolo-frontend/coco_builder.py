from datetime import datetime, timezone
from typing import Any, Callable
import uuid

import numpy as np

from config import IMAGE_SIZE
from utils import prepare_tensor_from_image_bytes


def coco_keypoints_from_detection(
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


def build_coco_keypoints_predictions_from_frames(
    extracted_frames: list[tuple[str, bytes]],
    *,
    infer_pose_fn: Callable[..., list[dict[str, Any]]],
    model_input_size: int = IMAGE_SIZE,
    det_conf_threshold: float = 0.10,
    kp_conf_threshold: float = 0.20,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for frame_name, frame_bytes in extracted_frames:
        image_tensor, image_width, image_height = prepare_tensor_from_image_bytes(
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

        detections = infer_pose_fn(image_tensor, conf_threshold=det_conf_threshold)
        for detection in detections:
            bbox_xywh = detection.get("bbox_xywh", [0.0, 0.0, 0.0, 0.0])
            if len(bbox_xywh) != 4:
                continue

            center_x, center_y, width, height = bbox_xywh
            x = (float(center_x) - float(width) / 2.0) * scale_x
            y = (float(center_y) - float(height) / 2.0) * scale_y
            width_scaled = float(width) * scale_x
            height_scaled = float(height) * scale_y

            coco_keypoints, num_keypoints = coco_keypoints_from_detection(
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


def generate_managed_dataset_name(prefix: str = "yolo_pose_frames") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"