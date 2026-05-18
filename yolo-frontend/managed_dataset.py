import json
from io import BytesIO
from typing import List, Tuple

import requests


def build_labels(extracted_frames: List[Tuple[str, bytes]]) -> List[str]:
    return ["unlabeled" for _ in extracted_frames]


def build_metadata(extracted_frames: List[Tuple[str, bytes]], dataset_name: str, *, source: str = "yolo-frontend") -> List[dict]:
    metadata = []
    for index, (frame_name, _bytes) in enumerate(extracted_frames):
        metadata.append(
            {
                "source": source,
                "label_status": "unlabeled",
                "dataset_name": dataset_name,
                "object_path": f"{dataset_name}/{frame_name}",
                "content_type": "image/png",
                "frame_name": frame_name,
                "frame_index": index,
            }
        )
    return metadata


def upload_managed_dataset(training_api_url: str, dataset_name: str, labels: List[str], metadata: List[dict], zip_bytes: bytes, timeout: int = 30):
    url = f"{training_api_url.rstrip('/')}/managed_datasets"
    response = requests.post(
        url,
        data={
            "dataset_name": dataset_name,
            "labels": json.dumps(labels),
            "metadata": json.dumps(metadata),
        },
        files={
            "file": ("video_frames.zip", zip_bytes, "application/zip"),
        },
        timeout=timeout,
    )
    return response
