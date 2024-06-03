import os
import shutil
from celery import Celery, shared_task
from celery.result import AsyncResult
from dataclasses import asdict, dataclass
from datetime import datetime
from flask import current_app, Flask
from tempfile import mkdtemp
from typing import Any, Optional
from uuid import uuid4

from app.errors import PluginNotFoundError
from app.plugin import PluginLoader, PluginInfo


@dataclass
class TaskResult:
    ready: bool
    successful: Optional[bool]
    value: any

    def as_dict(self):
        """Get a dictionary representation of the task result."""
        return asdict(self)


class WorkerManager:
    _plugin_loader: PluginLoader
    _app: Flask
    _celery_app: Celery

    @shared_task(ignore_result=False)
    def _run_task(self, *args, **kwargs) -> Optional[Any]:
        """Celery task for running a plugin.

        Parameters
        ----------
        *args : List[any]
            List of positional parameters.
        **kwargs : Dict[str, any]
            Dictionary of keyword arguments. The Kwargs dictionary should at least contain a
            'plugin_name' keyword parameter, which is used to select the correct plugin.
        """
        # Load the plugin
        if "plugin_name" not in kwargs:
            raise PluginNotFoundError()
        plugin_name = kwargs["plugin_name"]
        plugin_loader: PluginLoader = kwargs["plugin_loader"]
        plugin_loader.load_plugins()
        plugin = plugin_loader.get_plugin(plugin_name)
        if not plugin:
            raise PluginNotFoundError()

        # Generating the run name
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        run_name = f"{timestamp}_{plugin_name}_{uuid4().hex}"
        kwargs["run_name"] = run_name

        # Setting up temporary directory
        plugin_tmp_dir = mkdtemp(prefix=f"chimp_{run_name}")
        kwargs["temp_dir"] = plugin_tmp_dir

        # If datasets are provided, fetch the paths to the datasets
        if "datasets" in kwargs:
            datasets = {}
            for dataset_name, dataset_on_disk in kwargs["datasets"].items():
                dataset_dir = os.path.join(
                    current_app.config["DATA_DIRECTORY"], dataset_on_disk
                )
                datasets[dataset_name] = dataset_dir
            kwargs["datasets"] = datasets

        # Starting plugin
        print(f"Starting plugin '{plugin_name}' (directory: '{plugin_tmp_dir}')")
        run_id = plugin.run(*args, **kwargs)

        # Cleanup
        shutil.rmtree(plugin_tmp_dir)
        return run_id

    def init_app(
        self, app: Flask, plugin_loader: PluginLoader, celery_app: Celery
    ) -> None:
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.
        plugin_loader : PluginLoader
            A plugin loader instance that is used to load different plugins.
        celery_app : Celery
            The celery task queue integration to pass tasks to the background workers.
        """
        if "worker_manager" in app.extensions:
            raise RuntimeError(
                "A 'WorkerManager' instance has already been registered on this Flask app."
            )  # pragma: no cover
        app.extensions["worker_manager"] = self
        self._app = app
        self._plugin_loader = plugin_loader
        self._celery_app = celery_app

    def start_task(self, plugin_name: str, *args, **kwargs) -> Optional[str]:
        """Start a task given a specified plugin.

        Parameters
        ----------
        plugin_name : str
            Name of the plugin to run.
        *args : List[any]
            List of positional parameters to pass to the task.
        **kwargs : Dict[str, any]
            Dictionary of keyword arguments to pass to the task. Note that the 'plugin_name' keyword
            will be overwritten.

        Returns
        -------
        The ID of the task or None if the plugin could not be found
        """
        if self._plugin_loader.get_plugin(plugin_name) is None:
            return None
        kwargs["plugin_name"] = plugin_name
        res = self._run_task.delay(plugin_name, *args, **kwargs)
        return res.id

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get the PluginInfo for a given plugin by name.

        Parameters
        ----------
        plugin_name : str
            The name of the plugin to get the info for.

        Returns
        -------
        The PluginInfo object or None if the plugin could not be found.
        """
        plugin = self._plugin_loader.get_plugin(plugin_name)
        if plugin:
            return plugin.info()
        return None

    @staticmethod
    def get_task_status(task_id: str) -> Optional[TaskResult]:
        """Poll the status of a task given a task ID.

        Parameters
        ----------
        task_id
            The ID of the task to search for.

        Returns
        -------
        A TaskResult containing whether the task is ready or not and the result of the task. If no task
        with the given ID is found, None is returned.
        """
        res = AsyncResult(task_id)
        if res:
            ready = res.ready()
            return TaskResult(
                ready=ready,
                successful=res.successful() if ready else None,
                value=res.get() if ready else res.result,
            )
