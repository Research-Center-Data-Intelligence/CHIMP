import pytest
from celery.app.task import Task

from app import worker
from app.errors import PluginNotFoundError
from app.worker import WorkerManager, TaskResult


class TestWorkerManager:
    """Tests for the worker manager class."""

    def test_run_task(self, worker_manager: WorkerManager, plugin: str, mocker):
        """Test the _run_task method."""

        def patched_file_func(*args, **kwargs):
            pass

        mocker.patch.object(worker.os, "mkdir", new=patched_file_func)
        mocker.patch.object(worker.shutil, "copytree", new=patched_file_func)
        mocker.patch.object(worker.shutil, "rmtree", new=patched_file_func)

        with pytest.raises(PluginNotFoundError):
            worker_manager._run_task(worker_manager)

        with pytest.raises(PluginNotFoundError):
            worker_manager._run_task(worker_manager, plugin_name="doesnotexist")

        worker_manager._run_task(
            worker_manager, plugin_name=plugin, datasets={"dataset": "TestingDataset"}
        )

    def test_start_task(self, worker_manager: WorkerManager, loaded_plugin: str):
        """Test the start_task method."""

        class PatchedTask:
            def delay(obj, *args, **kwargs):
                class ResObj:
                    id = "Test-ID"

                return ResObj()

        assert not worker_manager.start_task("doesnotexist")

        worker_manager._run_task = PatchedTask()
        assert worker_manager.start_task(loaded_plugin) == "Test-ID"

    def test_get_task_status(self, worker_manager: WorkerManager, monkeypatch):
        """Test the get_task_status method."""

        def mocked_async_result(task_id):
            class MockedAsyncResult:
                result = 5

                def ready(self) -> bool:
                    return True

                def successful(self) -> bool:
                    return True

                def get(self) -> int:
                    return 5

            if task_id == "Test-ID":
                return MockedAsyncResult()

        from app import worker

        monkeypatch.setattr(worker, "AsyncResult", mocked_async_result)

        assert not worker_manager.get_task_status("doesnotexist")

        res = worker_manager.get_task_status("Test-ID")
        assert type(res) is TaskResult
        assert res.ready
        assert res.successful
        assert res.value == 5
