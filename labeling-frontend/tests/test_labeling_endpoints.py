"""Integration tests for labeling-frontend Flask endpoints (IT-03, IT-04).

datastore_access now forwards everything to training-api over HTTP, so the
only thing to mock here is requests.request — no live services required.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add labeling-frontend root to path so imports of main / datastore_access work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def client():
    """Flask test client with the training-api HTTP boundary mocked."""
    import main as labeling_main

    labeling_main.app.config["TESTING"] = True
    with labeling_main.app.test_client() as c:
        yield c


def _fake_response(status_code=200, json_data=None, content=b"", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.headers = headers or {}
    resp.raise_for_status.side_effect = None
    return resp


# ---------------------------------------------------------------------------
# IT-04 — GET /api/datapoints/<id>  (fetch annotation details + source link)
# ---------------------------------------------------------------------------

class TestDatapointDetails:
    def test_returns_details_for_known_id(self, client):
        """IT-04: endpoint returns id, dataset, filename and object_path."""
        fake_datapoint = {
            "id": 42,
            "dataset_name": "my_dataset",
            "filename": "frame_001.png",
            "object_path": "my_dataset/frame_001.png",
            "metadata": {"dataset_name": "my_dataset"},
        }
        fake_resp = _fake_response(200, {"status": "success", "datapoint": fake_datapoint})

        with patch("requests.request", return_value=fake_resp):
            resp = client.get("/api/datapoints/42")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == 42
        assert data["dataset_name"] == "my_dataset"
        assert data["filename"] == "frame_001.png"

    def test_returns_404_for_unknown_id(self, client):
        """IT-04: endpoint returns 404 when datapoint does not exist."""
        fake_resp = _fake_response(404, {"status": "error", "message": "Datapoint not found"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.get("/api/datapoints/9999")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# IT-03 — POST /api/datapoints/<id>/label  (save correction + status)
# ---------------------------------------------------------------------------

_VALID_ANNOTATION = {
    "format": "coco_keypoints",
    "bbox": [10, 20, 100, 200],
    "keypoints": [float(i) for i in range(51)],
}


class TestLabelDatapoint:
    def test_label_saves_annotation_and_returns_labeled_status(self, client):
        """IT-03: a successful training-api response is forwarded unchanged."""
        result = {
            "id": 1,
            "annotation": _VALID_ANNOTATION,
            "metadata": {"label_status": "labeled", "annotation_format": "coco_keypoints"},
        }
        fake_resp = _fake_response(200, {"status": "success", "result": result})

        with patch("requests.request", return_value=fake_resp) as mock_request:
            resp = client.post(
                "/api/datapoints/1/label",
                json={"annotation": _VALID_ANNOTATION},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == 1
        assert data["metadata"]["label_status"] == "labeled"

        # verify the connector forwarded the unwrapped annotation to training-api
        call_args, call_kwargs = mock_request.call_args
        assert call_args[0] == "POST"
        assert call_args[1].endswith("/labeling/datapoint/1/label")
        assert call_kwargs["json"] == _VALID_ANNOTATION

    def test_label_returns_404_for_unknown_id(self, client):
        """IT-03: returns 404 when datapoint does not exist."""
        fake_resp = _fake_response(404, {"status": "error", "message": "Datapoint not found"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.post("/api/datapoints/999/label", json={"annotation": _VALID_ANNOTATION})

        assert resp.status_code == 404

    def test_label_rejects_missing_annotation(self, client):
        """IT-03: returns 400 when training-api rejects an empty body."""
        fake_resp = _fake_response(400, {"status": "error", "message": "Request body must be JSON"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.post("/api/datapoints/1/label", json={})

        assert resp.status_code == 400

    def test_label_rejects_wrong_format(self, client):
        """IT-03: returns 400 when training-api rejects annotation.format."""
        bad = dict(_VALID_ANNOTATION, format="something_else")
        fake_resp = _fake_response(400, {"status": "error", "message": "annotation.format must be 'coco_keypoints'"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.post("/api/datapoints/1/label", json={"annotation": bad})

        assert resp.status_code == 400

    def test_label_rejects_wrong_keypoints_length(self, client):
        """IT-03: returns 400 when training-api rejects a malformed keypoints array."""
        bad = dict(_VALID_ANNOTATION, keypoints=[0.0] * 48)
        fake_resp = _fake_response(400, {"status": "error", "message": "annotation.keypoints must contain 51 values"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.post("/api/datapoints/1/label", json={"annotation": bad})

        assert resp.status_code == 400

    def test_label_rejects_wrong_bbox_length(self, client):
        """IT-03: returns 400 when training-api rejects a malformed bbox array."""
        bad = dict(_VALID_ANNOTATION, bbox=[1, 2, 3])
        fake_resp = _fake_response(400, {"status": "error", "message": "annotation.bbox must contain 4 values"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.post("/api/datapoints/1/label", json={"annotation": bad})

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# IT-04 — GET /api/datapoints/<id>/image  (fetch source image bytes)
# ---------------------------------------------------------------------------

class TestDatapointImage:
    def test_returns_image_bytes_for_known_id(self, client):
        """IT-04: endpoint streams image bytes forwarded from training-api."""
        fake_image = b"\x89PNG\r\n\x1a\nfake"
        fake_resp = _fake_response(
            200,
            content=fake_image,
            headers={"content-type": "image/png", "x-filename": "img.png"},
        )

        with patch("requests.request", return_value=fake_resp):
            resp = client.get("/api/datapoints/7/image")

        assert resp.status_code == 200
        assert resp.data == fake_image

    def test_returns_404_for_unknown_id(self, client):
        """IT-04: returns 404 when datapoint is not in DB."""
        fake_resp = _fake_response(404, {"status": "error", "message": "Datapoint not found"})

        with patch("requests.request", return_value=fake_resp):
            resp = client.get("/api/datapoints/999/image")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# IT-05 — POST /api/retrain  (trigger fine-tuning via training-api)
# ---------------------------------------------------------------------------

class TestRetrain:
    def test_triggers_retrain_on_training_server(self, client):
        """IT-05: endpoint forwards request to training-api and returns task_id."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"task_id": "abc-123", "status": "task started successfully"}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            resp = client.post(
                "/api/retrain",
                json={"dataset_name": "my_dataset"},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "triggered"
        assert data["training_response"]["task_id"] == "abc-123"

        # Verify the request was forwarded to the right endpoint
        call_args = mock_post.call_args
        assert "/tasks/run/YOLO+Pose" in call_args[0][0]
        assert call_args[1]["data"]["dataset_name"] == "my_dataset"

    def test_retrain_without_dataset_name_still_triggers(self, client):
        """IT-05: retrain without dataset_name triggers export-only mode."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"task_id": "xyz-456"}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            resp = client.post("/api/retrain", json={})

        assert resp.status_code == 200
        call_data = mock_post.call_args[1]["data"]
        assert "dataset_name" not in call_data

    def test_retrain_returns_500_when_training_server_unreachable(self, client):
        """IT-05: returns 500 when connection to training-api fails."""
        with patch("requests.post", side_effect=ConnectionError("refused")):
            resp = client.post("/api/retrain", json={"dataset_name": "ds"})

        assert resp.status_code == 500
        assert "error" in resp.get_json()
