from flask.testing import FlaskClient

from app.worker import WorkerManager, TaskResult


class TestTrainingEndpoints:
    """Tests for the training endpoints."""

    def test_get_plugins(self, client: FlaskClient):
        """Test the get_plugins endpoint."""
        resp = client.get("/plugins?reload_plugins=true")
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "status" in data and data["status"] == "successfully retrieved plugins"
        assert "reloaded plugins" in data and data["reloaded plugins"]
        assert "plugins" in data
        plugin_names = [p["name"] for p in data["plugins"]]
        assert "Example Plugin" in plugin_names and "Example 2 Plugin" in plugin_names

    def test_start_task(self, client: FlaskClient, mocker):
        """Test the start task endpoint."""

        def patched_start_task(obj, plugin_name: str, *args, **kwargs):
            if plugin_name == "Example Plugin" or plugin_name == "Example 2 Plugin":
                return "Testing-ID"

        mocker.patch.object(WorkerManager, "start_task", new=patched_start_task)

        # Successful flow
        resp = client.post(
            "/tasks/run/Example+2+Plugin",
            data={"datasets": '{"dataset": "TestingDataset"}', "start_value": 42},
        )
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert (
            "status" in data
            and data["status"]
            == "task started successfully, use '/tasks/poll/Testing-ID' to poll for the current status"
        )
        assert "task_id" in data and data["task_id"] == "Testing-ID"

        # Non existing plugin
        resp = client.post("/tasks/run/Plugin+Does+Not+Exist")
        assert resp.status_code == 404

        # Missing arguments
        resp = client.post(
            "/tasks/run/Example+2+Plugin",
            data={"datasets": '{"dataset": "TestingDataset"}'},
        )
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "Missing required argument 'start_value'"

        # No dataset
        resp = client.post("/tasks/run/Example+2+Plugin")
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "Must specify the required datasets"

        # Missing dataset
        resp = client.post(
            "/tasks/run/Example+2+Plugin",
            data={"datasets": '{"optional_ds": "TestingDataset"}'},
        )
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "Missing required dataset 'dataset'"

        # Non existing dataset
        resp = client.post(
            "/tasks/run/Example+2+Plugin",
            data={"datasets": '{"dataset": "DoesNotExist"}'},
        )
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "Dataset DoesNotExist not found"

        # Wrong formatted dataset JSON
        resp = client.post(
            "/tasks/run/Example+2+Plugin", data={"datasets": "{wrong: format: json}"}
        )
        assert resp.status_code == 400
        assert resp.get_json()["message"].startswith(
            "Could not decode the datasets dictionary:"
        )

    def test_poll_task(self, client: FlaskClient, mocker):
        """Test the poll task endpoint."""

        def patched_get_task_status(obj, task_id: str):
            if task_id == "Testing-ID":
                return TaskResult(ready=True, successful=True, value=5)

        mocker.patch.object(
            WorkerManager, "get_task_status", new=patched_get_task_status
        )

        resp = client.get("/tasks/poll/Testing-ID")
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "ready" in data and data["ready"]
        assert "successful" in data and data["successful"]
        assert "value" in data and data["value"] == 5

        resp = client.get("/tasks/poll/DOESNOTEXIST")
        assert resp.status_code == 404
