from flask import Flask
from typing import Union

from app.endpoints import health_endpoints
from app.errors import bp as errors_bp
from app.extensions import cors, plugin_loader


def create_app(config_obj: Union[str, object] = "app.config") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_obj)

    if not app.config["TESTING"]:
        # Setup logging
        pass

    # Register blueprints
    app.register_blueprint(errors_bp)
    app.register_blueprint(health_endpoints.bp)

    # Initialize extensions
    cors.init_app(app)
    plugin_loader.init_app(app)
    plugin_loader.load_plugins()
    print(plugin_loader.loaded_plugins(include_details=True))

    return app
