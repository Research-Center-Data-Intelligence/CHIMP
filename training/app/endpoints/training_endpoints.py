import json
import warnings
from datetime import datetime
from flask import abort, Blueprint, current_app, request, Request
from typing import Dict
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from app.endpoints.dataset_endpoints import upload_dataset
from app.plugin import PluginLoader
from app.worker import WorkerManager
#from redis import Redis




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
    plugin_info = plugin_loader.loaded_plugins(include_details=include_details)
    return {
        "status": "successfully retrieved plugins",
        "reloaded plugins": reload_plugins,
        "plugins": plugin_info,
    }


@bp.route("/tasks/run/<plugin_name>", methods=["POST"])
def start_task(plugin_name: str):
    """Run a task.

    Parameters
    ----------
    plugin_name : str
        The name of the plugin to run for the task
    (form data) : input parameters for the plugin from the url
    

    Returns
    -------
    A JSON object containing the status of the task and a task ID.

    Examples
    --------
    curl
        `curl -X POST -F "dataset=Example" http://localhost:5253/tasks/run/Example+Plugin`
    """

    # Check if plugin exists and retrieve the plugin info
    plugin_name = plugin_name.replace("+", " ")
    worker_manager: WorkerManager = current_app.extensions["worker_manager"]
    plugin_info = worker_manager.get_plugin_info(plugin_name)
    if not plugin_info:
        abort(404)

    #MV TODO: what request arguments to check?
    
    kwargs: Dict[str, any] = {}

    # Check the arguments
    for expected_args in plugin_info["arguments"].values():
        key = expected_args["name"]
        value = request.form.get(key)
        if not value:
            value = request.args.get(key)
        if not value and not expected_args.get("optional"):
            raise BadRequest(f"Missing required argument '{key}'")
        kwargs[key] = value

    print(kwargs)
    task_id = worker_manager.start_task(plugin_name, **kwargs)
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


"""
@bp.route("/labeling_tasks")
def get_labeling_tasks():
    print("Connecting to Redis...")
    redis_client = Redis(host="message-queue", port=6379, decode_responses=True)

    print("Fetching labeling task keys from Redis...")
    task_keys = redis_client.keys("labeling_task:*")
    print(f"Found task keys: {task_keys}")
    tasks = []

    for key in task_keys:
        try:
            print(f"Processing key: {key}")
            data = json.loads(redis_client.get(key))
            print(f"Data for {key}: {data}")
            dataset_id = key.split(":", 1)[1]
            user = dataset_id.split("_")[1] if "_" in dataset_id else "unknown"
            task = {
                "dataset_id": dataset_id,
                "total_images": data.get("total", 0),
                "labeled_percentage": 0,  
                "received": data.get("timestamp", "onbekend"),
                "user": user
            }
            print(f"Task constructed: {task}")
            tasks.append(task)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            current_app.logger.warning(f"Kon Redis-entry {key} niet verwerken: {e}")

    print(f"Returning tasks: {tasks}")
    return {"tasks": tasks}
"""





