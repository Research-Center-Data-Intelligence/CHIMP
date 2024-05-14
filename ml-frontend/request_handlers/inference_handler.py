from os import environ
import logging
import requests
from flask_socketio import SocketIO, emit
from flask import request,jsonify

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

    # Process the image (Example: Saving locally)
    image_path = f"images/{user_id}_image.jpg"
    with open(image_path, "wb") as image_file:
        image_file.write(image_blob)

    # Prepare data for sending to Flask backend
    payload = {
        'user_id': user_id,
        'image_blob': image_blob.decode('utf-8')  # Convert bytes to string for JSON serialization
    }

    # Make a POST request to your Flask backend's REST API
    backend_url = 'http://your-flask-backend-url/api/save_image'
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(backend_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        backend_data = response.json()
        # Emit an update using SocketIO or return data as needed
        emit('update-data', backend_data)
        return jsonify(success=True, message='Image processed and saved successfully')
    except requests.exceptions.RequestException as e:
        return jsonify(success=False, message=f'Error sending image data to backend: {str(e)}')



def add_as_websocket_handler(socket_io: SocketIO):
    global _on_connect, _on_disconnect, _process_image

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_image = socket_io.on('process-image')(_process_image)

    return socket_io
