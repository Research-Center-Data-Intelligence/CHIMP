import os
import shutil
from flask import Blueprint, current_app, request, Request
from tempfile import mkdtemp
from werkzeug.exceptions import BadRequest
from zipfile import ZipFile, BadZipFile
import re
from werkzeug.utils import secure_filename
import io
import zipfile
import json

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
    datastore = current_app.extensions["datastore"]
    return {
        "status": "successfully retrieved datasets",
        "datasets": [
            ds.replace("/", "")
            for ds in datastore.list_from_datastore("", recursive=False)
        ],
    }


@bp.route("/managed_datasets", methods=["POST"])
def upload_managed_dataset(passed_request: Request = None):
    """Upload a dataset from zip file. The datapoints in the zip should have list of labels (string) and a list of metadata (json / dict)

    Parameters
    ----------
    passed_request : Request
        A overwrite to support the (depricated) /model/train and /model/calibrate endpoints

    Returns
    -------
    Whether or not the upload was successful

    Examples
    --------
    curl
        `curl -X POST -F "file=@/path/to/zipfile.zip" -F "dataset_name=Example" http://localhost:5253/managed_datasets`
        `curl -X POST -F "file=@C:/CHIMP-data/calib_test_tiny.zip" -F "dataset_name=tiny_test" -F "labels=[\"angry\", \"disgusted\", \"disgusted\", \"neutral\"]"  -F "metadata=[{\"exp\":\"tinytest\",\"usr\":\"maarten\"},{\"exp\":\"tinytest\",\"usr\":\"maarten\"},{\"exp\":\"tinytest\",\"usr\":\"maarten\"},{\"exp\":\"tinytest\",\"usr\":\"maarten\"}]" http://localhost:5000/managed_datasets`
    """
    current_request = request
    if passed_request:
        current_request = passed_request  # pragma: no cover

    labels = current_request.form.get("labels")
    if not labels:
        raise BadRequest("No labels in request")
    try:
        labels = json.loads(labels)  
        if not isinstance(labels, list):
            raise BadRequest("Labels should be a list of strings")
    except json.JSONDecodeError:
        raise BadRequest("Invalid format for labels")


    metadata = current_request.form.get("metadata")
    if not metadata:
        raise BadRequest("No metadata in request")
    try:
        metadata = json.loads(metadata)  
        if not isinstance(metadata, list):
            raise BadRequest("Metadata should be a list of dictionaries")
    except json.JSONDecodeError:
        raise BadRequest("Invalid format for metadata")


    if "file" not in current_request.files:
        raise BadRequest("No file in request")
    file = current_request.files["file"]

    if not file.filename.endswith(".zip"):
        raise BadRequest("File should be a zip")

    dataset_name = current_request.form.get("dataset_name")
    if not dataset_name:
        raise BadRequest("Dataset name ('dataset_name') field missing")
    invalid_chars = re.compile(r'[<>:"/\\|?*]')
    if invalid_chars.search(dataset_name):
        raise BadRequest(
            "Dataset name ('dataset_name') should only contain characters allowed in path strings"
        )
    datastore = current_app.extensions["datastore"]
    if dataset_name in [
        ds.replace("/", "") for ds in datastore.list_from_datastore("", recursive=False)
    ]:
        raise BadRequest(f"Dataset with name '{dataset_name}' already exists")

    file_data = file.read()
    zip_buffer = io.BytesIO(file_data)

    with zipfile.ZipFile(zip_buffer, 'r') as zip_archive:
        file_names = zip_archive.namelist()
        num_files = len(file_names)

        if len(labels) != num_files:
            raise BadRequest(f"Number of labels ({len(labels)}) does not match number of files ({num_files})")
        
        if len(metadata) != num_files:
            raise BadRequest(f"Number of metadata entries ({len(metadata)}) does not match number of files ({num_files})")

        for i, file_name in enumerate(file_names):
            with zip_archive.open(file_name) as extracted_file:

                object_name = secure_filename(file_name)
                
                file_content = extracted_file.read()
                file_stream = io.BytesIO(file_content)
                file_stream.seek(0)  # Ensure pointer is at the start
                print("start uploading")
                datastore.store_object(dataset_name, file_stream, labels[i], metadata[i], object_name)
                print("done uploading")

    return {"status": "successfully uploaded dataset"}




@bp.route("/datasets", methods=["POST"])
def upload_dataset(passed_request: Request = None):
    """Upload a dataset as a zip file, which is made available for training.

    Parameters
    ----------
    passed_request : Request
        A overwrite to support the (depricated) /model/train and /model/calibrate endpoints

    Returns
    -------
    Whether or not the upload was successful

    Examples
    --------
    curl
        `curl -X POST -F "file=@/path/to/zipfile.zip" -F "dataset_name=Example" http://localhost:5253/datasets`
    """
    current_request = request
    if passed_request:
        current_request = passed_request  # pragma: no cover

    if "file" not in current_request.files:
        raise BadRequest("No file in request")
    file = current_request.files["file"]

    print(current_request.files)
    print(file.filename)
    print(file.filename.endswith(".zip"))


    if not file.filename.endswith(".zip"):
        raise BadRequest("File should be a zip")

    dataset_name = current_request.form.get("dataset_name")
    if not dataset_name:
        raise BadRequest("Dataset name ('dataset_name') field missing")
    invalid_chars = re.compile(r'[<>:"/\\|?*]')
    if invalid_chars.search(dataset_name):
        raise BadRequest(
            "Dataset name ('dataset_name') should only contain characters allowed in path strings"
        )
    datastore = current_app.extensions["datastore"]
    if dataset_name in [
        ds.replace("/", "") for ds in datastore.list_from_datastore("", recursive=False)
    ]:
        raise BadRequest(f"Dataset with name '{dataset_name}' already exists")

    tmpdir = mkdtemp(prefix="chimp_")
    zip_path = os.path.join(tmpdir, file.filename)
    file.save(zip_path)
    upload_path = os.path.join(tmpdir, "to_upload")
    os.mkdir(upload_path)

    try:
        with ZipFile(zip_path, "r") as f:
            f.extractall(upload_path)
    except BadZipFile:
        raise BadRequest("Invalid zip file")

    print("start uploading")
    datastore.store_file_or_folder(dataset_name, upload_path)
    print("done uploading")

    shutil.rmtree(tmpdir)
    return {"status": "successfully uploaded dataset"}