import os
import numpy as np
import cv2
from os import environ
import logging
import requests
import json
import zipfile
import re
from PIL import Image

from flask_socketio import SocketIO, emit
from flask import request, jsonify
from datetime import datetime
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from logic.image_processor import ImageProcessor
from io import BytesIO
#import psycopg2



import imageio.v3 as iio

INFERENCE_INTERVAL = 0

_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))
_image_processors: dict = {}


def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')
    _image_processors[request.sid] = ImageProcessor(INFERENCE_INTERVAL)


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')
    if request.sid in _image_processors:
        del _image_processors[request.sid]


def _process_image(data):
    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    image_blob = data['image_blob']

    img_processor = _image_processors.get(user_id, ImageProcessor(INFERENCE_INTERVAL))
    img_processor.load_image(image_blob)
    img_processor.process(user_id)

    data_to_emit = {'predictions': img_processor.predictions, 'status': img_processor.status_msg}

    emit('update-data', data_to_emit)

    return img_processor.get_image_blob()


def sanitize_timestamp(timestamp):
    return timestamp.replace("T", "_").replace(":", "-").replace(".", "-")


def _upload_managed_calibration_data(data):
    print("Processing calibration video blobs for upload to managed dataset")

    cascade_path = os.path.join(os.getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")
    upload_url = f"{TRAINING_SERVER_URL}/managed_datasets"
 
    user_id = data.get("user_id") or request.sid
    username = data["username"]
    video_blobs = data["image_blobs"]
    emotions = data["emotions"]
    timestamps = data["timestamps"]

    labels = []
    metadata = []
    zip_buffer = BytesIO()

    timestamp_group = sanitize_timestamp(datetime.now().isoformat())
    dataset_name = re.sub(r'[<>:"/\\|?*]', '', f"calibration_{username}_{timestamp_group}_{user_id}")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for video_blob, emotion, timestamp in zip(video_blobs, emotions, timestamps):
            sanitized_ts = sanitize_timestamp(timestamp)
            video_stream = BytesIO(video_blob)
            video_array = iio.imread(video_stream, plugin="pyav")

            for i, frame in enumerate(video_array):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) != 1:
                    continue

                for (x, y, w, h) in faces:
                    face_img = cv2.resize(gray[y:y + h, x:x + w], (96, 96))
                    img_pil = Image.fromarray(face_img)
                    img_buffer = BytesIO()
                    img_pil.save(img_buffer, format="PNG")
                    img_buffer.seek(0)

                    filename = f"img_{emotion}_{sanitized_ts}_{i:04d}.png"
                    zip_path = os.path.join("train", emotion, filename).replace("\\", "/")
                    zipf.writestr(zip_path, img_buffer.getvalue())

                    labels.append(emotion)
                    metadata.append({
                        "exp": "emotion_recognition",
                        "user": username,
                        "userid": user_id,
                        "timestamp": sanitized_ts,
                        "filename": filename
                    })

    zip_buffer.seek(0)

    files = {
        "file": (dataset_name + ".zip", zip_buffer.getvalue(), "application/zip"),
        "dataset_name": (None, dataset_name),
        "labels": (None, json.dumps(labels)),
        "metadata": (None, json.dumps(metadata))
    }

    print(f"[INFO] Uploading dataset '{dataset_name}' with {len(labels)} images...")
    response = requests.post(upload_url, files=files)

    try:
        response_data = response.json()
    except Exception:
        response_data = {"error": "Invalid JSON response from server"}

    if response.status_code != 200:
        print("[ERROR]", response_data)
        return response_data, response.status_code

    print(f"[SUCCESS] Uploaded dataset '{dataset_name}' successfully.")
    return response_data, response.status_code


def _upload_managed_pool_data(data):
    print("Processing pool video blobs for upload to managed dataset")

    cascade_path = os.path.join(os.getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")
    upload_url = f"{TRAINING_SERVER_URL}/managed_datasets"
    plugin_url = f"{TRAINING_SERVER_URL}/tasks/run/Active+Learning"
    EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")

    user_id = data.get("user_id") or request.sid
    username = data["username"]
    video_blobs = data["image_blobs"]
    timestamps = data["timestamps"]

    labels = []
    metadata = []
    zip_buffer = BytesIO()

    
    timestamp_group = sanitize_timestamp(datetime.now().isoformat())
    dataset_name = re.sub(r'[<>:"/\\|?*]', '', f"pool_{username}_{timestamp_group}_{user_id}")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for video_blob, raw_timestamp in zip(video_blobs, timestamps):
            
            sanitized_ts = sanitize_timestamp(raw_timestamp)  
            video_stream = BytesIO(video_blob)
            video_array = iio.imread(video_stream, plugin="pyav")

            for i, frame in enumerate(video_array):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) != 1:
                    continue

                for (x, y, w, h) in faces:
                    face_img = cv2.resize(gray[y:y + h, x:x + w], (96, 96))
                    img_pil = Image.fromarray(face_img)
                    img_buffer = BytesIO()
                    img_pil.save(img_buffer, format="PNG")
                    img_buffer.seek(0)

                    filename = f"img_pool_{sanitized_ts}_{i:04d}.png"
                    zip_path = os.path.join("pool", "unlabeled", filename).replace("\\", "/")
                    zipf.writestr(zip_path, img_buffer.getvalue())

                    labels.append("unlabeled")
                    metadata.append({
                        "exp": "emotion_recognition",
                        "user": username,
                        "userid": user_id,
                        "timestamp": raw_timestamp,  
                        "filename": filename,
                        "type": "pool"
                    })

    zip_buffer.seek(0)

    files = {
        "file": (dataset_name + ".zip", zip_buffer.getvalue(), "application/zip"),
        "dataset_name": (None, dataset_name),
        "labels": (None, json.dumps(labels)),
        "metadata": (None, json.dumps(metadata))
    }

    print(f"[INFO] Uploading POOL dataset '{dataset_name}' with {len(labels)} images...")
    response = requests.post(upload_url, files=files)

    try:
        response_data = response.json()
    except Exception:
        response_data = {"error": "Invalid JSON response from server"}

    if response.status_code != 200:
        print("[ERROR]", response_data)
        return response_data, response.status_code

    print(f"[SUCCESS] Uploaded POOL dataset '{dataset_name}' successfully.")

    form = {
        "experiment_name": EXPERIMENT_NAME,
        "pool_dataset": dataset_name,
        "query_size": str(100)
    }

    print(f"[TASK] Triggering Active Learning plugin for dataset '{dataset_name}'...")
    try:
        task_response = requests.post(plugin_url, data=form)
        task_json = task_response.json()
        print(f"[TASK RESPONSE] Status {task_response.status_code} | Response: {task_json}")
    except Exception as e:
        print(f"[ERROR] Failed to trigger Active Learning plugin: {e}")
        return {"error": f"Failed to trigger plugin: {str(e)}"}, 500

    return {
        "status": "Dataset uploaded and plugin triggered",
        "upload_response": response_data,
        "plugin_response": task_json
    }, 200



def _process_video(data):
    print("Processing video blobs")
    cascade_file = os.path.join(os.getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    EXPERIMENT_NAME=environ.get("EXPERIMENT_NAME")
    PLUGIN_NAME="Emotion+Recognition"
    TRAINING_SERVER_URL=environ.get("TRAINING_SERVER_URL")
    url = TRAINING_SERVER_URL + "/datasets"

    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    username =data['username']
    video_blobs = data['image_blobs']
    emotions = data['emotions']
    timestamps = data['timestamps']
    
    # send to datastore
    # Create a BytesIO object to hold the zip file in memory, don't write to disk
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:    
        for video_blob, emotion, timestamp in zip(video_blobs, emotions, timestamps):
            timestamp = sanitize_timestamp(timestamp)
            
            video_stream = BytesIO(video_blob)
            video_stream.seek(0)
            video_array = iio.imread(video_stream, plugin='pyav')

            recording_id = f"{username}_{emotion}_{timestamp}_{user_id}_recording"
            cnt=0
            for i, img in enumerate(video_array):
                # Get gray-scale version of the image, detect each face, and get for each face an emotion prediction.
                grey_frame = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                faces = face_cascade.detectMultiScale(grey_frame, 1.3, 5)
                if len(faces) != 1:
                    continue #only process if exactly one face is detected, other cases not supported
                else:
                    for index, (x, y, width, height) in enumerate(faces):
                        image = cv2.resize(grey_frame[y:y+height, x:x+width], (96, 96))

                        image = Image.fromarray(image.astype('uint8'))
                        # Save the image to a BytesIO buffer
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        buffer.seek(0)
                        
                        #define "file" name and data
                        name = os.path.join(f'img_{emotion}_{i:04d}.png')
                        zip_path = os.path.join(os.path.join("train", emotion), name)
                        zipf.writestr(zip_path, buffer.getvalue())
                        cnt=cnt+1

            print("processing blob with emotion ", emotion, " detected 1 face in nframe: ", cnt)
    
    zip_buffer.seek(0)

    #clean the dataset name as non alphanumeric characters are not allowed by the training and minio modules
    clean_id = re.sub(r'[<>:"/\\|?*]', '', f"calibration_{username}_{timestamp}_{user_id}")
    zip_buffer.name=clean_id
    files = {}
    files["file"] = (clean_id + '.zip', zip_buffer.getvalue(), 'application/zip')

    print("Sending image zip to dataset_name: ", clean_id)

    response = requests.request('POST',  url=url, data={"dataset_name" : clean_id}, files=files)
    print(response.json())
    if response.status_code!=200:
        return response.json(), response.status_code
        #raise BadRequest("Could not upload dataset zip")

    form = dict()
    form["calibration_id"] = username + '_' + user_id
    form["calibrate"] = True
    form["experiment_name"] = EXPERIMENT_NAME
    form["datasets"] = json.dumps({"train": "fer2013", "calibration" : clean_id})

    print("Requesting model calibrations: ", form)

    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME
    response = requests.request('POST',  url=url, data=form)
    print(response.json())
    return response.json(), response.status_code


def _label_image(data):
    """
    Handles image labeling events received via Socket.
    Receives dataset name, filename, and emotion label from the client,
    sends a labeling request to the backend, and emits a response back to the client.
    """
    # Log the receipt of a labeling event via Socket.IO
    print("[INFO] Received labeling event via Socket.IO:", data)

    try:
        # Extract required fields from the incoming data
        dataset_id = data["dataset_name"]
        filename = data["filename"]
        label = data["emotion"]

        # Check that all required fields are present
        if not all([dataset_id, filename, label]):
            emit("label_image_response", {"error": "Missing field(s)"}, room=request.sid)
            return

        # Prepare the URL for the backend labeling endpoint
        TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")
        url = f"{TRAINING_SERVER_URL}/label_image"

        # Prepare the form data for the POST request
        form = {
            "dataset_name": dataset_id,
            "filename": filename,
            "emotion": label
        }

        # Send the labeling request to the backend
        response = requests.post(url, data=form)
        response_data = response.json()

        # Emit a response back to the client based on the backend response
        if response.status_code == 200:
            emit("label_image_response", {"status": "ok", "filename": filename}, room=request.sid)
        else:
            emit("label_image_response", {"error": response_data.get("error", "Unknown error")}, room=request.sid)

    except Exception as e:
        # Handle and log any exceptions that occur
        print("[ERROR] during _label_image:", str(e))
        emit("label_image_response", {"error": str(e)}, room=request.sid)



def _train():
    PLUGIN_NAME="Emotion+Recognition"
    
    EXPERIMENT_NAME=environ.get("EXPERIMENT_NAME")
    TRAINING_SERVER_URL=environ.get("TRAINING_SERVER_URL")
    datasets=json.dumps({"train": "emotions"})
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME

    response = requests.request('POST',  url=url, data={"datasets" : datasets, "experiment_name" : EXPERIMENT_NAME})

    return response.json(), response.status_code

def _calibrate():
    PLUGIN_NAME="Emotion+Recognition"
    
    EXPERIMENT_NAME=environ.get("EXPERIMENT_NAME")
    TRAINING_SERVER_URL=environ.get("TRAINING_SERVER_URL")
    
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME

    form = dict()

    # get user_id from request
    if "user_id" not in request.args:
        return BadRequest("No user specified.")
    
    user_id = request.args["user_id"]
    trainnew = request.args["trainnew"]
    basedata = request.args["basedata"]
    newdata = request.args["newdata"]
    personaldata = request.args["personaldata"]

    
    form["user_id"] = user_id
    form["trainnew"] = trainnew
    form["basedata"] = basedata
    form["newdata"] = newdata
    form["personaldata"] = personaldata
    form["experiment_name"] = EXPERIMENT_NAME

    #MV TODO: fill in the form correctly        
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME
    print(url, form)
    response = requests.request('POST',  url=url, data=form)

    return response.json(), response.status_code


def add_as_websocket_handler(socket_io: SocketIO, app):
    global _on_connect, _on_disconnect, _process_image, _process_video, _upload_managed_calibration_data, _upload_managed_pool_data, _label_image

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_video = socket_io.on('process-video')(_process_video)
    _process_image = socket_io.on('process-image')(_process_image)
    _label_image = socket_io.on('label_image')(_label_image)
    _upload_managed_calibration_data = socket_io.on('upload_managed_calibration_data')(_upload_managed_calibration_data)
    _upload_managed_pool_data = socket_io.on('upload_managed_pool_data')(_upload_managed_pool_data)
    


    app.route('/train', methods=['POST'])(_train)
    app.route('/calibrate', methods=['POST'])(_calibrate)

    return socket_io
