from dotenv import load_dotenv
import os
import sys

from flask_socketio import SocketIO
from flask import Flask, render_template

from utils.logging_config import configure_logging

from request_handlers import inference_handler

basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, "..")))

app = Flask(__name__)
socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)

socket_io = inference_handler.add_as_websocket_handler(socket_io)

configure_logging(app)


@app.route('/')
def index():
    return render_template('index.html')


def run_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252, debug=True)

def get_app():
    load_dotenv()
    return app


if __name__ == '__main__':
    load_dotenv()
    run_app()
