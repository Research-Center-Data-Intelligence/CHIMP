import os
import shutil
from datetime import datetime
from flask import abort, Blueprint, current_app, request
from tempfile import mkdtemp
from werkzeug.exceptions import BadRequest

from app.plugin import PluginLoader
from app.worker import WorkerManager

bp = Blueprint("training", __name__)


@bp.route("/plugins")
def get_plugins():
    """Get a list of available plugins.

    Parameters
    ----------
    (optional query param) include_details : bool
        Whether or not to include details for each loaded plugin.
    (optional query param) reload_plugins : bool
        Whether or not to reload all plugins before generating a list of available plugins.

    Returns
    -------
    json:
        A json object containing a list of loaded plugins

    Examples
    --------
    curl
        `curl http://localhost:5253/plugins`
    curl
        `curl http://localhost:5253/plugins?include_details=true`
    curl
        `curl http://localhost:5253/plugins?reload_plugins=true`
    """
    include_details = request.args.get("include_details", type=bool, default=False)
    reload_plugins = request.args.get("reload_plugins", type=bool, default=False)

    plugin_loader: PluginLoader = current_app.extensions["plugin_loader"]
    if reload_plugins:
        plugin_loader.load_plugins()
    return {
        "status": "successfully retrieved plugins",
        "reloaded plugins": reload_plugins,
        "plugins": plugin_loader.loaded_plugins(include_details=include_details),
    }


@bp.route("/tasks/run/<plugin_name>", methods=["POST"])
def start_task(plugin_name: str):
    """Run a task.


    Parameters
    ----------
    plugin_name : str
        The name of the plugin to run for the task
    (form data) dataset : str
        The name of the dataset to use


    Returns
    -------
    A JSON object containing the status of the task and a task ID.

    Examples
    --------
    curl
        `curl -X POST -F "dataset=Example" http://localhost:5253/tasks/run/Example+Plugin`
    """
    dataset = request.form.get("dataset")
    if not dataset:
        raise BadRequest("Must specify a dataset")
    if dataset not in os.listdir(current_app.config["DATA_DIRECTORY"]):
        raise BadRequest(f"Dataset {dataset} not found")

    data_dir = os.path.join(current_app.config["DATA_DIRECTORY"], dataset)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    temp_data_dir = mkdtemp(prefix=f"chimp_{timestamp}_{dataset}_")
    temp_data_dir = os.path.join(temp_data_dir, dataset)
    shutil.copytree(data_dir, temp_data_dir)

    plugin_name = plugin_name.replace("+", " ")
    worker_manager: WorkerManager = current_app.extensions["worker_manager"]
    task_id = worker_manager.start_task(plugin_name, data_dir=temp_data_dir)
    if not task_id:
        abort(404)
    return {
        "status": f"task started successfully, use '/tasks/poll/{task_id}' to poll for the current status",
        "task_id": task_id,
    }


@bp.route("/tasks/poll/<task_id>")
def poll(task_id: str):
    """Poll the status of a task.

    Parameters
    ----------
    task_id : str
        ID of the task to check the status for.

    Returns
    -------
    The status of the task or a 404 error if no task with the given ID is found.

    Examples
    curl
        `curl http://localhost:5253/tasks/poll/aef0ff97-2f59-4ea2-9ce8-bd29c6a69637`
    """
    worker_manager: WorkerManager = current_app.extensions["worker_manager"]
    task_info = worker_manager.get_task_status(task_id)
    if not task_info:
        abort(404)
    return task_info.as_dict()
