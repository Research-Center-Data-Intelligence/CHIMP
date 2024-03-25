from flask import Flask
from typing import Union

from .endpoints import health_endpoints, inference_endpoints
from .errors import bp as errors_bp
from .extensions import cors, inference_manager


def create_app(config_obj: Union[str, object] = "src.config") -> Flask:
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
    inference_manager.init_app(app)

    return app
