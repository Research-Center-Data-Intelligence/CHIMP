from typing import Any

import numpy as np
import requests

from config import (
    MODEL_NAME,
    MODEL_SESSION_ID,
    MODEL_STAGE,
    REQUEST_TIMEOUT_SECONDS,
    SERVING_API_URL,
)
from utils import decode_yolo_pose_output


def infer_pose_tensor(
    model_input: np.ndarray,
    conf_threshold: float = 0.10,
    iou_threshold: float = 0.45,
    *,
    session: Any | None = None,
    serving_api_url: str = SERVING_API_URL,
    model_name: str = MODEL_NAME,
    model_stage: str = MODEL_STAGE,
    model_session_id: str = MODEL_SESSION_ID,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
):
    infer_url = f"{serving_api_url}/model/{model_name}/infer"
    params = {"stage": model_stage}
    if model_session_id:
        params["id"] = model_session_id

    payload = {"inputs": model_input.tolist()}
    http_client = session or requests
    response = http_client.post(
        infer_url,
        params=params,
        json=payload,
        timeout=timeout_seconds,
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
    return decode_yolo_pose_output(
        first_output,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )