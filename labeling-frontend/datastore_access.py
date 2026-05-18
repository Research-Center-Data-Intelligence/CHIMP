import json
import os
from typing import Any

import psycopg2
from minio import Minio
from psycopg2.extras import Json, RealDictCursor


DATASTORE_URI = os.environ.get("DATASTORE_URI", "datastore:9000")
DATASTORE_ACCESS_KEY = os.environ.get("DATASTORE_ACCESS_KEY", "")
DATASTORE_SECRET_KEY = os.environ.get("DATASTORE_SECRET_KEY", "")
DATASTORE_BUCKET = os.environ.get("DATASTORE_BUCKET", "manageddataset")

DATABASE_URI = os.environ.get("DATABASE_URI", "postgres-db")
DATABASE_USER = os.environ.get("DATABASE_USER", "chimp_user")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "chimp_password")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "chimp_database")

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


def _object_path_from_row(x_value: Any, metadata: dict | None) -> str | None:
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


def guess_mimetype(filename: str | None) -> str:
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


def list_unlabeled_datapoints(limit: int = 50) -> list[dict]:
    limit = max(1, min(limit or 50, 500))

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

    return datapoints


def get_datapoint_image(datapoint_id: int) -> tuple[bytes, str, str]:
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
        raise LookupError("Datapoint not found")

    metadata = row.get("metadata")
    object_path = _object_path_from_row(row.get("x"), metadata)
    if not object_path:
        raise ValueError("Datapoint has no resolvable object_path")

    filename = None
    content_type = None
    if isinstance(metadata, dict):
        filename = metadata.get("frame_name") or metadata.get("file_name")
        content_type = metadata.get("content_type")
    if not filename:
        filename = object_path.split("/", maxsplit=1)[-1]
    if not isinstance(content_type, str) or not content_type:
        content_type = guess_mimetype(filename)

    client = _get_minio_client()
    response = client.get_object(DATASTORE_BUCKET, object_path)
    try:
        image_bytes = response.read()
    finally:
        response.close()
        response.release_conn()

    return image_bytes, content_type, filename


def get_datapoint_details(datapoint_id: int) -> dict:
    conn = _get_db_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT id, x, y, metadata
            FROM datapoints
            WHERE id = %s
            LIMIT 1
            """,
            (datapoint_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise LookupError("Datapoint not found")

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    object_path = _object_path_from_row(row.get("x"), metadata)

    dataset_name = metadata.get("dataset_name") if isinstance(metadata, dict) else None
    filename = None
    if isinstance(metadata, dict):
      filename = metadata.get("frame_name") or metadata.get("file_name")

    return {
        "id": row.get("id"),
        "dataset_name": dataset_name,
        "filename": filename,
        "object_path": object_path,
        "metadata": metadata,
    }


def label_datapoint(datapoint_id: int, annotation: dict) -> dict:
    if not isinstance(annotation, dict):
        raise ValueError("annotation must be a JSON object")

    if annotation.get("format") != "coco_keypoints":
        raise ValueError("annotation.format must be 'coco_keypoints'")

    keypoints = annotation.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != 51:
        raise ValueError("annotation.keypoints must contain 51 values for 17 joints")

    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("annotation.bbox must contain 4 values [x, y, w, h]")

    conn = _get_db_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT id, metadata
            FROM datapoints
            WHERE id = %s
            LIMIT 1
            """,
            (datapoint_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise LookupError("Datapoint not found")

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    updated_metadata = dict(metadata)
    updated_metadata["label_status"] = "labeled"
    updated_metadata["annotation_format"] = "coco_keypoints"
    # compute actual number of labeled keypoints from the keypoints array (every 3rd value is v)
    try:
        kp = keypoints
        num_labeled = 0
        for i in range(2, len(kp), 3):
            if isinstance(kp[i], (int, float)) and kp[i] > 0:
                num_labeled += 1
    except Exception:
        num_labeled = 0
    updated_metadata["num_keypoints"] = num_labeled

    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE datapoints
            SET y = %s,
                metadata = %s
            WHERE id = %s
            """,
            (json.dumps(annotation, separators=(",", ":")), Json(updated_metadata), datapoint_id),
        )
    conn.commit()

    return {
        "id": datapoint_id,
        "annotation": annotation,
        "metadata": updated_metadata,
    }
