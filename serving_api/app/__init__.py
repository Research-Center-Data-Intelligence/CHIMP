from flask import Flask
from typing import Union

from app.endpoints import health_endpoints, inference_endpoints
from app.errors import bp as errors_bp
from app.extensions import cors, inference_manager, connector


def create_app(config_obj: Union[str, object] = "app.config") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_obj)

    if not app.config["TESTING"]:
        # Setup logging
        pass

    # Register blueprints
    app.register_blueprint(errors_bp)
    app.register_blueprint(health_endpoints.bp)
    app.register_blueprint(inference_endpoints.bp)

    # Initialize extensions
    cors.init_app(app)
    connector.init_app(app, app.config["TRACKING_URI"])
    inference_manager.init_app(app, connector)

    return app
