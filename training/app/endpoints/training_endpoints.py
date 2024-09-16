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
def start_task(plugin_name: str, passed_request=None):
    """Run a task.


    Parameters
    ----------
    plugin_name : str
        The name of the plugin to run for the task
    (form data) dataset : str
        The name of the dataset to use
    passed_request : Request
        A overwrite to support the (depricated) /model/train and /model/calibrate endpoints


    Returns
    -------
    A JSON object containing the status of the task and a task ID.

    Examples
    --------
    curl
        `curl -X POST -F "dataset=Example" http://localhost:5253/tasks/run/Example+Plugin`
    """
    # This code is required to support the deprecated /model/train and /model/calibrate endpoints
    # until they are removed
    current_request = request
    if passed_request:
        current_request = passed_request  # pragma: no cover

    # Check if plugin exists and retrieve the plugin info
    plugin_name = plugin_name.replace("+", " ")
    worker_manager: WorkerManager = current_app.extensions["worker_manager"]
    plugin_info = worker_manager.get_plugin_info(plugin_name)
    if not plugin_info:
        abort(404)

    # Check datasets
    datasets_kwarg = {}
    if plugin_info["datasets"]:
        datasets = current_request.form.get("datasets")
        # If datasets are expected, but not provided
        if not datasets:
            raise BadRequest("Must specify the required datasets")
        try:
            datasets = json.loads(datasets)
        except json.JSONDecodeError as ex:
            raise BadRequest(f"Could not decode the datasets dictionary: {ex.msg}")

        # Check if all datasets are provided
        datastore = current_app.extensions["datastore"]
        datasets_on_datastore = [
            ds.replace("/", "")
            for ds in datastore.list_from_datastore("", recursive=False)
        ]
        for dataset_name, dataset_info in plugin_info["datasets"].items():
            if dataset_name not in datasets:
                # If a dataset is not provided, check if it is optional
                if "optional" not in dataset_info or not dataset_info["optional"]:
                    raise BadRequest(f"Missing required dataset '{dataset_name}'")
            else:
                dataset_name_on_datastore = datasets[dataset_name]
                # Check if the specified dataset is available on the datastore
                if dataset_name_on_datastore not in datasets_on_datastore:
                    raise BadRequest(f"Dataset {dataset_name_on_datastore} not found")

                datasets_kwarg[dataset_name] = dataset_name_on_datastore
    kwargs: Dict[str, any] = {}
    if datasets_kwarg:
        kwargs["datasets"] = datasets_kwarg

    # Check the arguments
    for expected_args in plugin_info["arguments"].values():
        key = expected_args["name"]
        value = request.form.get(key)
        if not value:
            value = request.args.get(key)
        if not value and not expected_args.get("optional"):
            raise BadRequest(f"Missing required argument '{key}'")
        kwargs[key] = value

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


@bp.route("/model/<train_or_calibrate>", methods=["POST"])
def model_train_or_calibrate(train_or_calibrate: str):  # pragma: no cover
    """Legacy route to support the old style of inference.

    WARNING: Calling the /model/train endpoint is deprecated, use the /tasks/run/<plugin> endpoint instead
    """
    warnings.warn(
        "Calling the /model/train endpoint is deprecated, use the /tasks/run/<plugin> endpoint instead"
    )

    class ReplacementRequest:
        form: Dict
        files: Dict

        def __init__(self, request_files, request_form):
            self.files = request_files
            self.form = request_form

    form = {"datasets": {"train": current_app.config["LEGACY_DATASET_NAME"]}}
    files = {}

    if train_or_calibrate == "calibrate":
        form["calibrate"] = True
        # get user_id from request
        if "user_id" not in request.args:
            return BadRequest("No user specified.")
        user_id = request.args["user_id"]
        form["calibration_id"] = user_id

        # get zipfile from request
        if len(request.files) == 0:
            return BadRequest("No files uploaded.")
        if "zipfile" not in request.files:
            return BadRequest("Different file expected.")
        file = request.files["zipfile"]
        if file.filename == "":
            return BadRequest("No file selected.")
        if not (
            "." in file.filename and file.filename.rsplit(".", 1)[1].lower() == "zip"
        ):
            return BadRequest("File type not allowed. Must be a zip.")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        dataset_name = secure_filename(
            f"emotions{user_id}{timestamp}".replace("-", "").replace("_", "")
        )
        files["file"] = file
        form["dataset_name"] = dataset_name
        form["datasets"]["calibration"] = dataset_name
        form["datasets"] = json.dumps(form["datasets"])
        try:
            upload_dataset(ReplacementRequest(files, form))
        except BadRequest as ex:
            print(ex)
            raise BadRequest("Could not store dataset")
    else:
        form["datasets"] = json.dumps(form["datasets"])
    return start_task(
        current_app.config["LEGACY_PLUGIN_NAME"], ReplacementRequest(files, form)
    )
