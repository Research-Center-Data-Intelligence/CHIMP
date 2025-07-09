import os
import shutil
from flask import Blueprint, current_app, request, Request, jsonify
from tempfile import mkdtemp
from werkzeug.exceptions import BadRequest
from zipfile import ZipFile, BadZipFile
import re
from werkzeug.utils import secure_filename
import io
import zipfile
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

bp = Blueprint("dataset", __name__)
REDIS_HOST = "message-queue"
REDIS_PORT = 6379
REDIS_QUEUE_NAME = "labeled_image_queue"



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
                
                metadata[i]["used_in_training"] = False

                file_content = extracted_file.read()
                file_stream = io.BytesIO(file_content)
                file_stream.seek(0)  # Ensure pointer is at the start
                datastore.store_object(dataset_name, file_stream, labels[i], metadata[i], object_name)

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

@bp.route("/labeling_tasks", methods=["GET"])
def get_labeling_tasks():
    DB_CONFIG = {
        "dbname": os.getenv("DATABASE_NAME", "chimp_database"),
        "user": os.getenv("DATABASE_USER", "chimp_user"),
        "password": os.getenv("DATABASE_PASSWORD", "chimp_password"),
        "host": os.getenv("DATABASE_URI", "postgres-db"),
        "port": 5432
    }

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        lt.dataset_id,
                        dp.metadata ->> 'user' AS user,
                        dp.metadata ->> 'timestamp' AS timestamp,
                        lt.total_images,
                        lt.num_labeled,
                        lt.status
                    FROM labeling_tasks lt
                    LEFT JOIN LATERAL (
                        SELECT metadata
                        FROM datapoints
                        WHERE datapoints.metadata ->> 'exp' = 'emotion_recognition'
                        AND datapoints.metadata ->> 'type' = 'pool'
                        AND datapoints.x LIKE '%' || lt.dataset_id || '%'
                        ORDER BY id ASC
                        LIMIT 1
                    ) dp ON true
                    WHERE lt.status = 'pending'
                    ORDER BY lt.dataset_id DESC;
                """)

                rows = cursor.fetchall()

        return {"tasks": rows}

    except Exception as e:
        print("[ERROR] Error while fetching labeling_tasks:", e)
        return {"tasks": []}, 500
@bp.route("/labeling_task_data/<dataset_id>", methods=["GET"])
def get_labeling_task_data(dataset_id):
    try:
        datastore = current_app.extensions["datastore"]

        with datastore._db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT selection, total_images FROM labeling_tasks
                WHERE dataset_id = %s
                LIMIT 1
            """, (dataset_id,))
            result = cursor.fetchone()

        if not result:
            return jsonify({"error": "Labeling task not found"}), 404

        selected_filenames = result["selection"]
        total = result["total_images"]

        print("[DEBUG] selected files:", selected_filenames)

        images_data = {}
        for filename in selected_filenames:
            object_path = f"{dataset_id}/{filename}"
            try:
                print(f"[DEBUG] Fetching from MinIO: {object_path}")
                image_bytes = datastore._client.get_object("datasets", object_path).read()
                images_data[filename] = image_bytes.hex()
            except Exception as e:
                print(f"[WARNING] Could not fetch file: {object_path} | {e}")

        return jsonify({
            "images": images_data,
            "total_images": total,
            "num_labeled": total - len(selected_filenames) if selected_filenames else total
        })

    except Exception as e:
        print(f"[ERROR] Could not fetch labeling task data: {e}")
        return jsonify({"error": "Internal server error"}), 500

    

@bp.route("/label_image", methods=["POST"])
def label_image():
    try:
        dataset_name = request.form.get("dataset_name")
        filename = request.form.get("filename")
        emotion = request.form.get("emotion")

        if not dataset_name or not filename or not emotion:
            return jsonify({"error": "dataset_name, filename and emotion are required"}), 400

        datastore = current_app.extensions["datastore"]

        with datastore._db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id FROM datapoints
                WHERE x LIKE %s
                ORDER BY id ASC
                LIMIT 1
            """, (f"%{dataset_name}/{filename}",))
            result = cursor.fetchone()
        if not result:
            return jsonify({"error": "File not found in database"}), 404

        datapoint_id = result["id"]

        with datastore._db_conn.cursor() as cursor:
            cursor.execute("""
                UPDATE datapoints
                SET y = %s
                WHERE id = %s
            """, (emotion, datapoint_id))

            cursor.execute("""
                UPDATE labeling_tasks
                SET selection = (
                    SELECT jsonb_agg(elem)
                    FROM jsonb_array_elements_text(selection) AS elem
                    WHERE elem <> %s
                )
                WHERE dataset_id = %s
            """, (filename, dataset_name))

            cursor.execute("""
                SELECT selection, total_images FROM labeling_tasks
                WHERE dataset_id = %s
            """, (dataset_name,))
            selection_result = cursor.fetchone()
            selection = selection_result[0]
            total = selection_result[1]

            labeled_count = total - len(selection) if selection else total

            cursor.execute("""
                UPDATE labeling_tasks
                SET num_labeled = %s
                WHERE dataset_id = %s
            """, (labeled_count, dataset_name))

        datastore._db_conn.commit()

        task = {
            "datapoint_id": datapoint_id,
            "dataset_name": dataset_name,
            "filename": filename,
            "emotion": emotion
        }

        redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_client.rpush(REDIS_QUEUE_NAME, json.dumps(task))

        print(f"[INFO] Labeled and removed from task: {filename} → {emotion}")
        return jsonify({"status": "ok", "filename": filename, "emotion": emotion})

    except Exception as e:
        print(f"[ERROR] during label_image: {str(e)}")
        return jsonify({"error": str(e)}), 500


