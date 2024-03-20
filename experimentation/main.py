from dotenv import load_dotenv

from flask_cors import CORS
from flask import Flask, abort

from request_handlers import health_handler, experimentation_handler
import logging

import sys
import os

from messaging import MessagingLoggingHandler

basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, "..")))

app = Flask(__name__)
CORS(app)

app = health_handler.add_as_route_handler(app)
app = experimentation_handler.add_as_route_handler(app)

messaging_handler = MessagingLoggingHandler()
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.INFO)
werkzeug_logger.addHandler(messaging_handler)
socketio_logger = logging.getLogger("socketio")
socketio_logger.setLevel(logging.INFO)
socketio_logger.addHandler(messaging_handler)
engineio_logger = logging.getLogger("engineio")
engineio_logger.setLevel(logging.INFO)
engineio_logger.addHandler(messaging_handler)


@app.route("/")
def index():
    return abort(418)


def get_app():
    load_dotenv()
    return app


def main():
    # TODO: Add a security token?
    # Currently functioning as a webapi to build models, or to update existing models upon request. Endpoint variables
    #   for MLFlow are defined by environment variable.
    #   endpoint defined in environment variables.
    app.run(host="0.0.0.0", port=5253)


if __name__ == "__main__":
    load_dotenv()
    main()
