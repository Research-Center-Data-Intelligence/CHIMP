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
from flask import request
from datetime import datetime
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from logic.image_processor import ImageProcessor
from io import BytesIO

import imageio.v3 as iio

INFERENCE_INTERVAL = 0

_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))
_image_processors: dict = {}


def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')
    _image_processors[request.sid] = ImageProcessor(INFERENCE_INTERVAL)


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')
    print(request)
    # Note: has a vulnerability in which by not explicitly disconnecting from other side of the socket, more image
    #           processors keep getting cached
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
    form["datasets"] = json.dumps({"train": "emotions", "calibration" : clean_id})

    print("Requesting model calibrations: ", form)

    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME
    response = requests.request('POST',  url=url, data=form)
    print(response.json())
    return response.json(), response.status_code

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
    files = {}

    # get user_id from request
    if "user_id" not in request.args:
        return BadRequest("No user specified.")
    
    user_id = request.args["user_id"]
    form["calibration_id"] = user_id
    form["calibrate"] = True
    form["experiment_name"] = EXPERIMENT_NAME

    # get zipfile from request
    if len(request.files) == 0:
        return BadRequest("No files uploaded.")
    if "zipfile" not in request.files:
        return BadRequest("Different file expected.")
    file = request.files["zipfile"]
    if file.filename == "":
        return BadRequest("No file selected.")
    if not (
        "." in file.filename and file.filename.rsplit(".", 1)[1].lower() == "zip"
    ):
        return BadRequest("File type not allowed. Must be a zip.")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    calibration_dataset_name = secure_filename(
        f"calib_emotions{user_id}{timestamp}".replace("-", "").replace("_", "")
    )
    #convert to bytes?
    file_bytes = BytesIO()
    file.save(file_bytes)
    file_bytes.seek(0)
    file_bytes.name=file.filename
    files["file"] = (file_bytes.name, file_bytes, 'application/zip')
    
    form["datasets"] = json.dumps({"train": "emotions", "calibration" : calibration_dataset_name})
    print(calibration_dataset_name)
    url = TRAINING_SERVER_URL + "/datasets"
    response = requests.request('POST',  url=url, data={"dataset_name" : calibration_dataset_name}, files=files)
    if response.status_code!=200:
        return response.json(), response.status_code
        #raise BadRequest("Could not upload dataset zip")
        
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME
    response = requests.request('POST',  url=url, data=form)

    return response.json(), response.status_code


def add_as_websocket_handler(socket_io: SocketIO, app):
    global _on_connect, _on_disconnect, _process_image, _process_video

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_video = socket_io.on('process-video')(_process_video)
    _process_image = socket_io.on('process-image')(_process_image)

    app.route('/train', methods=['POST'])(_train)
    app.route('/calibrate', methods=['POST'])(_calibrate)

    return socket_io
