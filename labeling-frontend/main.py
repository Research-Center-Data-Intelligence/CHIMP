import os

import io

from flask import Flask, jsonify, render_template, request, send_file

from datastore_access import (
    get_datapoint_details,
    get_datapoint_image,
    label_datapoint,
    list_unlabeled_datapoints,
)
import requests
from datetime import datetime


PORT = int(os.environ.get("PORT", "5261"))

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("labeling.html")


@app.get("/annotate/<int:datapoint_id>")
def annotate(datapoint_id: int):
    return render_template("annotate.html", datapoint_id=datapoint_id)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/unlabeled")
def api_unlabeled():
    try:
        limit = request.args.get("limit", default=50, type=int)
        datapoints = list_unlabeled_datapoints(limit=limit)
        return jsonify({"datapoints": datapoints})
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to query unlabeled datapoints", "details": str(ex)}), 500


@app.get("/api/datapoints/<int:datapoint_id>/image")
def api_datapoint_image(datapoint_id: int):
    try:
        image_bytes, content_type, filename = get_datapoint_image(datapoint_id)
        return send_file(
            io.BytesIO(image_bytes),
            mimetype=content_type,
            as_attachment=False,
            download_name=filename,
        )
    except LookupError as ex:
        return jsonify({"error": str(ex)}), 404
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to fetch image", "details": str(ex)}), 500


@app.get("/api/datapoints/<int:datapoint_id>")
def api_datapoint_details(datapoint_id: int):
    try:
        return jsonify(get_datapoint_details(datapoint_id))
    except LookupError as ex:
        return jsonify({"error": str(ex)}), 404
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to fetch datapoint details", "details": str(ex)}), 500


@app.post("/api/datapoints/<int:datapoint_id>/label")
def api_datapoint_label(datapoint_id: int):
    payload = request.get_json(silent=True) or {}
    annotation = payload.get("annotation")

    try:
        result = label_datapoint(datapoint_id, annotation)
        return jsonify(result)
    except LookupError as ex:
        return jsonify({"error": str(ex)}), 404
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to label image", "details": str(ex)}), 500


@app.post("/api/retrain")
def api_retrain():
    """Trigger a retrain run on the training server for YOLO Pose.

    Expects JSON: { "dataset_name": "..." }
    """
    payload = request.get_json(silent=True) or {}
    dataset_name = payload.get("dataset_name")

    training_url = os.environ.get("TRAINING_SERVER_URL", "http://training-api:8000")
    plugin_path = "/tasks/run/YOLO+Pose"

    # Use fixed experiment name for aggregation, generate a unique run_name
    experiment_name = payload.get("experiment_name") or "yolo_pose_demo"
    run_name = payload.get("run_name") or f"{dataset_name or 'all'}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    form = {
        "experiment_name": experiment_name,
        "run_name": run_name,
    }
    if dataset_name:
        form["dataset_name"] = dataset_name

    try:
        r = requests.post(training_url + plugin_path, data=form, timeout=120)
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Failed to contact training server", "details": str(ex)}), 500

    try:
        data = r.json()
    except Exception:
        data = {"status_code": r.status_code, "text": r.text}

    return jsonify({"status": "triggered", "training_response": data}), (200 if r.status_code == 200 else 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
