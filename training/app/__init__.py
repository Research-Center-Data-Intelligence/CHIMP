import os.path

from celery import Celery, Task
from flask import Flask
from typing import Union

from app.endpoints import dataset_endpoints, health_endpoints, training_endpoints
from app.errors import bp as errors_bp
from app.extensions import cors, plugin_loader, worker_manager
from app.plugin import PluginLoader


def create_app(config_obj: Union[str, object] = "app.config") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_obj)

    if not app.config["TESTING"]:
        # Setup logging
        pass

    if not os.path.exists(app.config["DATA_DIRECTORY"]):
        os.mkdir(app.config["DATA_DIRECTORY"])

    # Register blueprints
    app.register_blueprint(errors_bp)
    app.register_blueprint(dataset_endpoints.bp)
    app.register_blueprint(health_endpoints.bp)
    app.register_blueprint(training_endpoints.bp)

    # Initialize extensions
    cors.init_app(app)
    plugin_loader.init_app(app)
    plugin_loader.load_plugins()
    print(plugin_loader.loaded_plugins(include_details=True))
    celery_app = create_celery_app(app)
    worker_manager.init_app(app, plugin_loader, celery_app)

    return app


def create_celery_app(app: Flask):
    _plugin_loader = app.extensions["plugin_loader"]

    class FlaskTask(Task):
        def __call__(
            self,
            *args: object,
            plugin_loader: PluginLoader = _plugin_loader,
            **kwargs: object
        ) -> object:
            with app.app_context():
                kwargs["plugin_loader"] = plugin_loader
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(
        dict(
            broker_url=app.config["CELERY_BROKER_URL"],
            result_backend=app.config["CELERY_RESULT_BACKEND"],
            task_ignore_result=True,
        )
    )
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app
