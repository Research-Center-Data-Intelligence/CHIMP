import os

import io

import psycopg2
from minio import Minio
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, render_template, request, send_file


PORT = int(os.environ.get("PORT", "5261"))
DATASTORE_URI = os.environ.get("DATASTORE_URI", "datastore:9000")
DATASTORE_ACCESS_KEY = os.environ.get("DATASTORE_ACCESS_KEY", "")
DATASTORE_SECRET_KEY = os.environ.get("DATASTORE_SECRET_KEY", "")
DATASTORE_BUCKET = os.environ.get("DATASTORE_BUCKET", "manageddataset")

DATABASE_URI = os.environ.get("DATABASE_URI", "postgres-db")
DATABASE_USER = os.environ.get("DATABASE_USER", "chimp_user")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "chimp_password")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "chimp_database")

app = Flask(__name__)


_db_conn = None
_minio_client = None


def _get_db_conn():
    global _db_conn  # noqa: PLW0603
    if _db_conn is None or getattr(_db_conn, "closed", 1):
        _db_conn = psycopg2.connect(
            dbname=DATABASE_NAME,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            host=DATABASE_URI,
            port=5432,
        )
    return _db_conn


def _get_minio_client() -> Minio:
    global _minio_client  # noqa: PLW0603
    if _minio_client is None:
        _minio_client = Minio(
            DATASTORE_URI,
            access_key=DATASTORE_ACCESS_KEY,
            secret_key=DATASTORE_SECRET_KEY,
            secure=False,
        )
    return _minio_client


def _object_path_from_row(x_value, metadata):
    if isinstance(metadata, dict):
        object_path = metadata.get("object_path")
        if isinstance(object_path, str) and object_path:
            return object_path

    if not isinstance(x_value, str) or not x_value:
        return None

    marker = f"/{DATASTORE_BUCKET}/"
    if marker in x_value:
        return x_value.split(marker, 1)[1]

    return None


def _guess_mimetype(filename: str | None) -> str:
    if not filename:
        return "application/octet-stream"
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


@app.get("/")
def index():
    return render_template("labeling.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/unlabeled")
def api_unlabeled():
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit or 50, 500))

    try:
        conn = _get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, x, y, metadata
                FROM datapoints
                WHERE metadata ->> 'label_status' = 'unlabeled'
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall() or []

        datapoints = []
        for row in rows:
            metadata = row.get("metadata")
            object_path = _object_path_from_row(row.get("x"), metadata)
            dataset_name = metadata.get("dataset_name") if isinstance(metadata, dict) else None
            filename = None
            if isinstance(metadata, dict):
                filename = metadata.get("frame_name") or metadata.get("file_name")
            if not filename and isinstance(object_path, str) and object_path:
                filename = object_path.split("/", maxsplit=1)[-1]

            datapoints.append(
                {
                    "id": row.get("id"),
                    "metadata": metadata,
                    "object_path": object_path,
                    "dataset_name": dataset_name,
                    "filename": filename,
                }
            )

        return jsonify({"datapoints": datapoints})
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to query unlabeled datapoints", "details": str(ex)}), 500


@app.get("/api/datapoints/<int:datapoint_id>/image")
def api_datapoint_image(datapoint_id: int):
    try:
        conn = _get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, x, metadata
                FROM datapoints
                WHERE id = %s
                LIMIT 1
                """,
                (datapoint_id,),
            )
            row = cursor.fetchone()

        if not row:
            return jsonify({"error": "Datapoint not found"}), 404

        metadata = row.get("metadata")
        object_path = _object_path_from_row(row.get("x"), metadata)
        if not object_path:
            return jsonify({"error": "Datapoint has no resolvable object_path"}), 400

        filename = None
        content_type = None
        if isinstance(metadata, dict):
            filename = metadata.get("frame_name") or metadata.get("file_name")
            content_type = metadata.get("content_type")
        if not filename:
            filename = object_path.split("/", maxsplit=1)[-1]
        if not isinstance(content_type, str) or not content_type:
            content_type = _guess_mimetype(filename)

        client = _get_minio_client()
        response = client.get_object(DATASTORE_BUCKET, object_path)
        try:
            image_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        return send_file(
            io.BytesIO(image_bytes),
            mimetype=content_type,
            as_attachment=False,
            download_name=filename,
        )
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to fetch image", "details": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
