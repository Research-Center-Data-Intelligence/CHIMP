from celery import Celery, shared_task
from celery.result import AsyncResult
from dataclasses import asdict, dataclass
from flask import Flask
from typing import Any, Optional

from app.errors import PluginNotFoundError
from app.plugin import PluginLoader


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
        if "plugin_name" not in kwargs:
            raise PluginNotFoundError()
        plugin_name = kwargs["plugin_name"]
        plugin_loader: PluginLoader = kwargs["plugin_loader"]
        plugin_loader.load_plugins()
        plugin = plugin_loader.get_plugin(kwargs["plugin_name"])
        if not plugin:
            plugin_loader.load_plugins()
            plugin = plugin_loader.get_plugin(plugin_name)
            if not plugin:
                raise PluginNotFoundError()
        return plugin.run(*args, **kwargs)

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
            )
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
