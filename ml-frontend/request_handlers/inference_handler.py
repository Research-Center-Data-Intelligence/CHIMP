import base64
from os import environ
import logging
from flask_socketio import SocketIO, emit
from flask import request

from logic.image_processor import ImageProcessor

INFERENCE_INTERVAL = 0

_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))
_image_processors: dict = {}


def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')
    _image_processors[request.sid] = ImageProcessor(INFERENCE_INTERVAL)


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')

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


def _process_video(data):
    print("TEST")
    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    image_blob = data['image_blob']
    emotion = data['emotion']
    
    video_data = base64.b64decode(image_blob)
    
    # Save vTESTideo data to a file
    print(f"{user_id}_{emotion}_recording.webm")
    with open(f"{user_id}_{emotion}_recording.webm", "wb") as video_file:
        video_file.write(video_data)
    
    _logger.debug(f'Saved video for user {user_id} with emotion {emotion}.')
    


def add_as_websocket_handler(socket_io: SocketIO):
    global _on_connect, _on_disconnect, _process_image, _process_video

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_image = socket_io.on('process-image')(_process_image)
    _process_video = socket_io.on('process-video')(_process_video)
    print("adding hand")
    return socket_io
