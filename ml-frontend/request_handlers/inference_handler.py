import os
import numpy as np
import cv2
from os import environ
import logging
import requests
import json

from flask_socketio import SocketIO, emit
from flask import request
from datetime import datetime
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from logic.image_processor import ImageProcessor
from io import BytesIO

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

    emit('update-data', img_processor.predictions)

    return img_processor.get_image_blob()


def sanitize_timestamp(timestamp):
    return timestamp.replace("T", "_").replace(":", "-").replace(".", "-")

  
def _process_video(data):
    print("TEST")
    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    username =data['username']
    video_blob = data['image_blob']
    emotion = data['emotion']
    timestamp = sanitize_timestamp(data['timestamp'])
    
    # Save the video Blob to a temporary file
    video_path = f"{username}_{emotion}_{timestamp}_{user_id}_recording.webm"
    with open(video_path, "wb") as video_file:
        video_file.write(video_blob)
    
    _logger.debug(f'Saved video for user {user_id} with emotion {emotion}.')
    print(f"Video path: {video_path}")

    # Display the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        cv2.imshow('Video Footage', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #numpyframes = np.asarray(allframes)
    cap.release()

    cv2.destroyAllWindows()
    os.remove(video_path)
   
    video_array = np.array(frames)
    #remove local storage in next commit --> instead send to datastore
    npy_path = f"{username}_{emotion}_{timestamp}_{user_id}_recording.npy"
    np.save(npy_path, video_array)
    print(f"NumPy array to {npy_path}")
    
    # TEMP
    np_array = np.load(npy_path)
    if np.array_equal(video_array, np_array):
        print("Successfully saved")
    else:
        print("Something went wrong")
    # END OF TEMP
  

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
    #conver to bytes?
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
    _process_image = socket_io.on('process-image')(_process_image)
    
    _process_video = socket_io.on('process-video')(_process_video)
    print("added process video")

    app.route('/train', methods=['POST'])(_train)
    app.route('/calibrate', methods=['POST'])(_calibrate)

    return socket_io
