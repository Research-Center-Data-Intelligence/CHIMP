from dotenv import load_dotenv

from flask_cors import CORS
from flask import Flask, abort

from request_handlers import health_handler, experimentation_handler
import logging


app = Flask(__name__)
CORS(app)

app = health_handler.add_as_route_handler(app)
app = experimentation_handler.add_as_route_handler(app)

logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.INFO)
logging.getLogger('engineio').setLevel(logging.INFO)


@app.route('/')
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
    app.run(host='0.0.0.0', port=7500)


if __name__ == '__main__':
    load_dotenv()
    main()
