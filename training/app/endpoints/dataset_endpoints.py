import os
import shutil
from flask import Blueprint, current_app, request
from tempfile import mkdtemp
from werkzeug.exceptions import BadRequest
from zipfile import ZipFile, BadZipFile

bp = Blueprint("dataset", __name__)


@bp.route("/datasets")
def get_datasets():
    """Get a list of available datasets.

    Returns
    -------
    A list of available datasets.

    Examples
    --------
    curl
        `curl http://localhost:5253/datasets`
    """
    dataset_path = current_app.config["DATA_DIRECTORY"]
    return {
        "status": "successfully retrieved datasets",
        "datasets": [folder for folder in os.listdir(dataset_path)],
    }


@bp.route("/datasets", methods=["POST"])
def upload_dataset():
    """Upload a dataset as a zip file, which is made available for training.

    Returns
    -------
    Whether or not the upload was successful

    Examples
    --------
    curl
        `curl -X POST -F "file=@/path/to/zipfile.zip" -F "dataset_name=Example" http://localhost:5253/datasets`
    """
    if "file" not in request.files:
        raise BadRequest("No file in request")
    file = request.files["file"]
    if not file.filename:
        raise BadRequest("File can not be empty")
    if not file.filename.endswith(".zip"):
        raise BadRequest("File should be a zip")

    dataset_name = request.form.get("dataset_name")
    if not dataset_name:
        raise BadRequest("Dataset name ('dataset_name') field missing")
    if not dataset_name.isalnum():
        raise BadRequest(
            "Dataset name ('dataset_name') should only contain alphanumeric characters"
        )

    tmpdir = mkdtemp(suffix="chimp_")
    zip_path = os.path.join(tmpdir, file.filename)
    file.save(zip_path)

    dataset_path = current_app.config["DATA_DIRECTORY"]
    try:
        with ZipFile(zip_path, "r") as f:
            f.extractall(os.path.join(dataset_path, dataset_name))
    except BadZipFile:
        raise BadRequest("Invalid zip file")

    shutil.rmtree(tmpdir)
    return {"status": "successfully uploaded dataset"}
