import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template, request, send_file
from managed_dataset import build_labels, build_metadata, upload_managed_dataset
from config import DATASET_NAME, DEFAULT_FRAME_COUNT, REQUEST_TIMEOUT_SECONDS, TRAINING_API_URL
from inference import infer_pose_tensor
from utils import prepare_tensor_from_data_url
from video_utils import extract_png_frames_from_video, frames_to_zip_bytes

app = Flask(__name__)
logger = logging.getLogger(__name__)
REQUEST_SESSION = requests.Session()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/pose-infer", methods=["POST"])
def pose_infer():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    body = request.get_json(silent=True) or {}
    image_data_url = body.get("image")
    if not isinstance(image_data_url, str) or not image_data_url:
        return jsonify({"error": "Missing image data URL in 'image' field"}), 400

    try:
        tensor = prepare_tensor_from_data_url(image_data_url)
        detections = infer_pose_tensor(tensor, session=REQUEST_SESSION)
        return jsonify({"detections": detections})
    except requests.HTTPError as ex:
        logger.exception("Serving API returned an error")
        return (
            jsonify(
                {
                    "error": "Serving API returned an error",
                    "details": str(ex),
                }
            ),
            502,
        )
    except requests.RequestException as ex:
        logger.exception("Could not reach serving API")
        return (
            jsonify(
                {
                    "error": "Could not reach serving API",
                    "details": str(ex),
                }
            ),
            502,
        )
    except Exception as ex:  # noqa: BLE001
        logger.exception("Inference failed")
        return jsonify({"error": "Inference failed", "details": str(ex)}), 500


@app.route("/api/video-to-png-frames", methods=["POST"])
def video_to_png_frames():
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "Missing video file in 'video' field"}), 400

    frame_count = request.form.get("frame_count", default=DEFAULT_FRAME_COUNT, type=int)
    if frame_count is None or frame_count <= 0:
        return jsonify({"error": "frame_count must be a positive integer"}), 400

    try:
        video_bytes = video_file.read()
        extracted_frames = extract_png_frames_from_video(video_bytes, frame_count=frame_count)
        output_zip = BytesIO(frames_to_zip_bytes(extracted_frames))
        return send_file(
            output_zip,
            mimetype="application/zip",
            as_attachment=True,
            download_name="video_frames.zip",
        )
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": "Video frame extraction failed", "details": str(ex)}), 500


@app.route("/api/video-to-managed-dataset", methods=["POST"])
def video_to_managed_dataset():
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "Missing video file in 'video' field"}), 400

    frame_count = request.form.get("frame_count", default=DEFAULT_FRAME_COUNT, type=int)
    if frame_count is None or frame_count <= 0:
        return jsonify({"error": "frame_count must be a positive integer"}), 400

    dataset_name = request.form.get("dataset_name", type=str) or DATASET_NAME

    try:
        video_bytes = video_file.read()
        extracted_frames = extract_png_frames_from_video(video_bytes, frame_count=frame_count)

        upload_batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        unique_extracted_frames = [
            (f"{upload_batch_id}_{frame_name}", frame_bytes)
            for frame_name, frame_bytes in extracted_frames
        ]

        output_zip = frames_to_zip_bytes(unique_extracted_frames)

        labels = build_labels(unique_extracted_frames)
        metadata = build_metadata(unique_extracted_frames, dataset_name)

        response = upload_managed_dataset(
            TRAINING_API_URL,
            dataset_name,
            labels,
            metadata,
            output_zip,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if not response.ok:
            details = None
            try:
                details = response.json()
            except Exception:  # noqa: BLE001
                details = response.text
            logger.error("Training API upload failed with status %s", response.status_code)
            return (
                jsonify(
                    {
                        "error": "Training API upload failed",
                        "status_code": response.status_code,
                        "details": details,
                    }
                ),
                502,
            )

        return jsonify(
            {
                "status": "uploaded",
                "dataset_name": dataset_name,
                "frame_count": len(unique_extracted_frames),
            }
        )
    except ValueError as ex:
        logger.exception("Validation error during managed dataset upload")
        return jsonify({"error": str(ex)}), 400
    except requests.RequestException as ex:
        logger.exception("Could not reach training API")
        return (
            jsonify({"error": "Could not reach training API", "details": str(ex)}),
            502,
        )
    except Exception as ex:  # noqa: BLE001
        logger.exception("Upload to managed dataset failed")
        return jsonify({"error": "Upload to managed dataset failed", "details": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5260, debug=True)
