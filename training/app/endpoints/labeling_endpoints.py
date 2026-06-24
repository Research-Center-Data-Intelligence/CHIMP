import json

from flask import Blueprint, current_app, request, send_file

bp = Blueprint("labeling", __name__)


@bp.route("/labeling/unlabeled_datapoints", methods=["GET"])
def list_unlabeled_datapoints():
    limit = request.args.get("limit", default=50, type=int)
    source = request.args.get("source", default="yolo-frontend")
    datastore = current_app.extensions["datastore"]
    # access ManagedMinioDatastore instance

    try:
        rows = datastore.get_unlabeled_datapoints(source, limit)
        datapoints = []
        for row in rows:
            metadata = row.get("metadata") or {}
            datapoints.append({
                "id": row.get("id"),
                "dataset_name": metadata.get("dataset_name"),
                "filename": metadata.get("frame_name"),
                "object_path": metadata.get("object_path"),
                "metadata": metadata,
            })
        return {"status": "success", "datapoints": datapoints}, 200
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>/image", methods=["GET"])
def get_datapoint_image(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    try:
        datapoint = datastore.get_datapoint(datapoint_id)
        metadata = datapoint.get("metadata") or {}
        object_path = metadata.get("object_path")
        if not object_path:
            return {"status": "error", "message": "Datapoint has no resolvable object_path"}, 400

        image_data = datastore.load_object_to_memory(object_path)
        if image_data is None:
            return {"status": "error", "message": "Image not found in storage"}, 404

        filename = metadata.get("frame_name")
        content_type = metadata.get("content_type")

        response = send_file(
            image_data,
            mimetype=content_type,
            as_attachment=False,
            download_name=filename,
        )
        response.headers["X-Filename"] = filename
        return response
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>", methods=["GET"])
def get_datapoint_details(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    try:
        datapoint = datastore.get_datapoint(datapoint_id)
        metadata = datapoint.get("metadata") or {}
        return {
            "status": "success",
            "datapoint": {
                "id": datapoint.get("id"),
                "dataset_name": metadata.get("dataset_name"),
                "filename": metadata.get("frame_name"),
                "object_path": metadata.get("object_path"),
                "metadata": metadata,
            },
        }, 200
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>/label", methods=["POST"])
def label_datapoint(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    annotation = request.get_json(silent=True)
    if not isinstance(annotation, dict):
        return {"status": "error", "message": "Request body must be a JSON object"}, 400

    if annotation.get("format") != "coco_keypoints":
        return {"status": "error", "message": "annotation.format must be 'coco_keypoints'"}, 400

    keypoints = annotation.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != 51:
        return {"status": "error", "message": "annotation.keypoints must contain 51 values for 17 joints"}, 400

    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"status": "error", "message": "annotation.bbox must contain 4 values [x, y, w, h]"}, 400

    try:
        datapoint = datastore.get_datapoint(datapoint_id)
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500

    updated_metadata = dict(datapoint.get("metadata") or {})
    updated_metadata["label_status"] = "labeled"
    updated_metadata["annotation_format"] = "coco_keypoints"

    num_labeled = 0
    for i in range(2, len(keypoints), 3):
        if isinstance(keypoints[i], (int, float)) and keypoints[i] > 0:
            num_labeled += 1
    updated_metadata["num_keypoints"] = num_labeled

    try:
        updated = datastore.update_object(datapoint_id, json.dumps(annotation), updated_metadata)
        return {
            "status": "success",
            "result": {
                "id": updated["id"],
                "annotation": annotation,
                "metadata": updated["metadata"],
            },
        }, 200
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500
