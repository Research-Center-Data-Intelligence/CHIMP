import io

from flask import Blueprint, current_app, request, send_file

bp = Blueprint("labeling", __name__)


@bp.route("/labeling/unlabeled_datapoints", methods=["GET"])
def list_unlabeled_datapoints():
    limit = request.args.get("limit", default=50, type=int)
    datastore = current_app.extensions["datastore"]
    # access ManagedMinioDatastore instance

    try:
        datapoints = datastore.get_unlabeled_datapoints(limit)
        return {"status": "success", "datapoints": datapoints}, 200
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>/image", methods=["GET"])
def get_datapoint_image(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    try:
        image_bytes, content_type, filename = datastore.get_datapoint_image(datapoint_id)
        response = send_file(
            io.BytesIO(image_bytes),
            mimetype=content_type,
            as_attachment=False,
            download_name=filename,
        )
        response.headers["X-Filename"] = filename
        return response
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>", methods=["GET"])
def get_datapoint_details(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    try:
        datapoint = datastore.get_datapoint_details(datapoint_id)
        return {"status": "success", "datapoint": datapoint}, 200
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500


@bp.route("/labeling/datapoint/<int:datapoint_id>/label", methods=["POST"])
def label_datapoint(datapoint_id: int):
    datastore = current_app.extensions["datastore"]

    annotation = request.get_json(silent=True)
    if not annotation:
        return {"status": "error", "message": "Request body must be JSON"}, 400

    try:
        result = datastore.label_datapoint(datapoint_id, annotation)
        return {"status": "success", "result": result}, 200
    except LookupError:
        return {"status": "error", "message": "Datapoint not found"}, 404
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}, 500
