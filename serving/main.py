import sys
import os

# This is required to make imports work consistently across different
# machines. This needs to be executed before other imports
basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, "..")))

import logging
from dotenv import load_dotenv
from flask import Flask, abort

from request_handlers import health_handler, inference_handler
from messaging import MessagingLoggingHandler

app = Flask(__name__)
app = health_handler.add_as_route_handler(app)
app = inference_handler.add_as_route_handler(app)

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
    # Currently functioning as middleware to invoke inference using the mlflow serving api to load a models from
    #   endpoint defined in environment variables.
    app.run(host="0.0.0.0", port=5254)


if __name__ == "__main__":
    load_dotenv()
    main()
