"""Integration tests for labeling-frontend Flask endpoints (IT-03, IT-04).

All database and MinIO calls are mocked — no live services required.
"""
import io
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Add labeling-frontend root to path so imports of main / datastore_access work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def client():
    """Flask test client with all external I/O mocked."""
    import datastore_access
    import main as labeling_main

    labeling_main.app.config["TESTING"] = True
    with labeling_main.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# IT-04 — GET /api/datapoints/<id>  (fetch annotation details + source link)
# ---------------------------------------------------------------------------

class TestDatapointDetails:
    def test_returns_details_for_known_id(self, client):
        """IT-04: endpoint returns id, dataset, filename and object_path."""
        fake_row = {
            "id": 42,
            "x": "http://minio/manageddataset/my_dataset/frame_001.png",
            "y": None,
            "metadata": {
                "dataset_name": "my_dataset",
                "frame_name": "frame_001.png",
                "label_status": "unlabeled",
            },
        }
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = fake_row

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("datastore_access._get_db_conn", return_value=mock_conn):
            resp = client.get("/api/datapoints/42")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == 42
        assert data["dataset_name"] == "my_dataset"
        assert data["filename"] == "frame_001.png"
        assert "frame_001.png" in (data.get("object_path") or "")

    def test_returns_404_for_unknown_id(self, client):
        """IT-04: endpoint returns 404 when datapoint does not exist."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("datastore_access._get_db_conn", return_value=mock_conn):
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
    def _make_mock_conn(self, existing_metadata=None):
        """Return a mock DB connection whose SELECT returns a row with given metadata."""
        meta = existing_metadata or {"label_status": "unlabeled"}
        fake_row = {"id": 1, "metadata": meta}

        select_cursor = MagicMock()
        select_cursor.__enter__ = lambda s: s
        select_cursor.__exit__ = MagicMock(return_value=False)
        select_cursor.fetchone.return_value = fake_row

        update_cursor = MagicMock()
        update_cursor.__enter__ = lambda s: s
        update_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = [select_cursor, update_cursor]
        return mock_conn

    def test_label_saves_annotation_and_returns_labeled_status(self, client):
        """IT-03: valid annotation is accepted, status changes to 'labeled'."""
        mock_conn = self._make_mock_conn()

        with patch("datastore_access._get_db_conn", return_value=mock_conn):
            resp = client.post(
                "/api/datapoints/1/label",
                json={"annotation": _VALID_ANNOTATION},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == 1
        assert data["annotation"]["format"] == "coco_keypoints"
        assert data["metadata"]["label_status"] == "labeled"
        assert data["metadata"]["annotation_format"] == "coco_keypoints"
        # verify DB UPDATE was committed
        mock_conn.commit.assert_called_once()

    def test_label_stores_num_keypoints(self, client):
        """IT-03: num_keypoints is computed from visibility flags and stored in metadata."""
        annotation = dict(_VALID_ANNOTATION)
        # set every 3rd value (visibility) to 2 for 5 keypoints, 0 for rest
        kps = [0.0] * 51
        for i in [2, 5, 8, 11, 14]:  # 5 keypoints with v=2
            kps[i] = 2.0
        annotation["keypoints"] = kps

        mock_conn = self._make_mock_conn()
        with patch("datastore_access._get_db_conn", return_value=mock_conn):
            resp = client.post("/api/datapoints/1/label", json={"annotation": annotation})

        assert resp.status_code == 200
        assert resp.get_json()["metadata"]["num_keypoints"] == 5

    def test_label_returns_404_for_unknown_id(self, client):
        """IT-03: returns 404 when datapoint does not exist."""
        cursor = MagicMock()
        cursor.__enter__ = lambda s: s
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor

        with patch("datastore_access._get_db_conn", return_value=mock_conn):
            resp = client.post("/api/datapoints/999/label", json={"annotation": _VALID_ANNOTATION})

        assert resp.status_code == 404

    def test_label_rejects_missing_annotation(self, client):
        """IT-03: returns 400 when annotation key is missing from payload."""
        resp = client.post("/api/datapoints/1/label", json={})
        assert resp.status_code == 400

    def test_label_rejects_wrong_format(self, client):
        """IT-03: returns 400 when annotation.format != 'coco_keypoints'."""
        bad = dict(_VALID_ANNOTATION, format="something_else")
        resp = client.post("/api/datapoints/1/label", json={"annotation": bad})
        assert resp.status_code == 400

    def test_label_rejects_wrong_keypoints_length(self, client):
        """IT-03: returns 400 when keypoints array has wrong length."""
        bad = dict(_VALID_ANNOTATION, keypoints=[0.0] * 48)
        resp = client.post("/api/datapoints/1/label", json={"annotation": bad})
        assert resp.status_code == 400

    def test_label_rejects_wrong_bbox_length(self, client):
        """IT-03: returns 400 when bbox array has wrong length."""
        bad = dict(_VALID_ANNOTATION, bbox=[1, 2, 3])
        resp = client.post("/api/datapoints/1/label", json={"annotation": bad})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# IT-04 — GET /api/datapoints/<id>/image  (fetch source image bytes)
# ---------------------------------------------------------------------------

class TestDatapointImage:
    def test_returns_image_bytes_for_known_id(self, client):
        """IT-04: endpoint streams image bytes from MinIO."""
        fake_row = {
            "id": 7,
            "x": "http://minio/manageddataset/ds/img.png",
            "metadata": {"object_path": "ds/img.png", "content_type": "image/png"},
        }
        cursor = MagicMock()
        cursor.__enter__ = lambda s: s
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = fake_row

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor

        fake_image = b"\x89PNG\r\n\x1a\nfake"
        mock_response = MagicMock()
        mock_response.read.return_value = fake_image
        mock_minio = MagicMock()
        mock_minio.get_object.return_value = mock_response

        with patch("datastore_access._get_db_conn", return_value=mock_conn), \
             patch("datastore_access._get_minio_client", return_value=mock_minio):
            resp = client.get("/api/datapoints/7/image")

        assert resp.status_code == 200
        assert resp.data == fake_image

    def test_returns_404_for_unknown_id(self, client):
        """IT-04: returns 404 when datapoint is not in DB."""
        cursor = MagicMock()
        cursor.__enter__ = lambda s: s
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor

        with patch("datastore_access._get_db_conn", return_value=mock_conn):
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
