import os

import requests

TRAINING_SERVER_URL = os.environ.get("TRAINING_SERVER_URL", "http://training-api:8000")


def _request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{TRAINING_SERVER_URL}{path}", **kwargs)

    if response.status_code == 404:
        raise LookupError(response.json().get("message", "Not found"))
    if response.status_code == 400:
        raise ValueError(response.json().get("message", "Bad request"))
    response.raise_for_status()

    return response


def list_unlabeled_datapoints(limit: int = 50) -> list[dict]:
    response = _request(
        "GET", "/labeling/unlabeled_datapoints", params={"limit": limit}, timeout=10
    )
    return response.json().get("datapoints", [])


def get_datapoint_image(datapoint_id: int) -> tuple[bytes, str, str]:
    response = _request(
        "GET", f"/labeling/datapoint/{datapoint_id}/image", timeout=30
    )
    content_type = response.headers.get("content-type", "application/octet-stream")
    filename = response.headers.get("x-filename", "image")
    return response.content, content_type, filename


def get_datapoint_details(datapoint_id: int) -> dict:
    response = _request("GET", f"/labeling/datapoint/{datapoint_id}", timeout=10)
    return response.json().get("datapoint", {})


def label_datapoint(datapoint_id: int, annotation: dict) -> dict:
    response = _request(
        "POST", f"/labeling/datapoint/{datapoint_id}/label", json=annotation, timeout=10
    )
    return response.json().get("result", {})
