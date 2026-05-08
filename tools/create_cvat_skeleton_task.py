#!/usr/bin/env python3
"""Create a CVAT task with a COCO-style skeleton label and upload images.

Usage:
  python tools/create_cvat_skeleton_task.py \
    --cvat-url http://localhost:8088 \
    --token <CVAT_API_TOKEN> \
    --image-dir /path/to/images \
    --task-name "COCO Keypoint Task" \
    [--project-id 4]

This uses the CVAT REST API directly and creates a label with type 'skeleton' and
sublabels for the 17 COCO keypoints. It then uploads all images from the
`--image-dir` and finalizes the upload.
"""
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import sys
from io import BytesIO
from typing import Iterable

import requests

COCO_KEYPOINTS = [
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
]

COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]

# Simple placeholder SVG (normalized coords) to hint CVAT how to connect points.
# This does not need precise coords; it only helps the UI show bones.
DEFAULT_SKELETON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200">'
    + ''.join(
        f'<line x1="{10 + i*5}" y1="{10 + i*6}" x2="{12 + j*5}" y2="{12 + j*6}" stroke="#000" stroke-width="1"/>'
        for i, j in ((a, b) for a, b in COCO_SKELETON[:10])
    )
    + '</svg>'
)


def _iter_images(directory: str) -> Iterable[str]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
        for p in glob.glob(os.path.join(directory, ext)):
            yield p


def _cvat_request(session: requests.Session, base_url: str, method: str, path: str, **kwargs):
    url = base_url.rstrip("/") + path
    r = session.request(method=method, url=url, timeout=120, **kwargs)
    return r


def create_task(session: requests.Session, base_url: str, name: str, project_id: int | None = None) -> int:
    payload = {"name": name, "labels": [
        {
            "name": "person",
            "type": "skeleton",
            "attributes": [],
            "sublabels": [{"name": kp, "type": "points", "attributes": []} for kp in COCO_KEYPOINTS],
            "svg": DEFAULT_SKELETON_SVG,
        }
    ]}
    if project_id is not None:
        payload["project_id"] = int(project_id)

    resp = _cvat_request(session, base_url, "POST", "/api/tasks", json=payload)
    if resp.status_code >= 400:
        try:
            details = json.dumps(resp.json(), indent=2)
        except Exception:
            details = resp.text
        raise RuntimeError(f"Failed to create task (status={resp.status_code}): {details}")
    return int(resp.json()["id"])


def upload_images(session: requests.Session, base_url: str, task_id: int, image_paths: list[str]) -> str | None:
    endpoint = f"/api/tasks/{task_id}/data/"
    start_resp = _cvat_request(session, base_url, "POST", endpoint, headers={"Upload-Start": "1"})
    if start_resp.status_code not in (200, 202):
        raise RuntimeError(f"Upload start failed: {start_resp.status_code} {start_resp.text}")

    files = []
    opened = []
    try:
        for i, path in enumerate(image_paths):
            f = open(path, "rb")
            opened.append(f)
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            files.append((f"client_files[{i}]", (os.path.basename(path), f, mime)))

        multiple_resp = _cvat_request(
            session,
            base_url,
            "POST",
            endpoint,
            headers={"Upload-Multiple": "1"},
            data={"image_quality": "100"},
            files=files,
        )
        if multiple_resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Upload multiple failed: {multiple_resp.status_code} {multiple_resp.text}")
    finally:
        for f in opened:
            f.close()

    finish_resp = _cvat_request(
        session,
        base_url,
        "POST",
        endpoint,
        headers={"Upload-Finish": "1"},
        json={"image_quality": 100, "sorting_method": "lexicographical"},
    )
    if finish_resp.status_code >= 400:
        raise RuntimeError(f"Upload finish failed: {finish_resp.status_code} {finish_resp.text}")

    payload = finish_resp.json() if finish_resp.content else {}
    return payload.get("rq_id")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cvat-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--task-name", default="COCO Keypoint Task")
    p.add_argument("--project-id", type=int, default=None)
    args = p.parse_args(argv)

    image_dir = args.image_dir
    image_paths = list(_iter_images(image_dir))
    if not image_paths:
        print("No image files found in", image_dir)
        return 2

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {args.token}"})

    print("Creating task...")
    task_id = create_task(session, args.cvat_url, args.task_name, project_id=args.project_id)
    print("Task created:", task_id)

    print(f"Uploading {len(image_paths)} images...")
    rq_id = upload_images(session, args.cvat_url, task_id, image_paths)
    print("Upload request id:", rq_id)
    print("Task URL:", f"{args.cvat_url.rstrip('/')}/tasks/{task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
