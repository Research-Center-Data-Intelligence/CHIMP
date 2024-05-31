import os
import numpy as np
import pytest
import shutil
from flask import Flask
from flask.testing import FlaskClient
from sklearn import svm
from tempfile import mkdtemp

from app import connectors
from app.connectors import BaseConnector
from app.plugin import BasePlugin, PluginLoader
from app.worker import WorkerManager


@pytest.fixture
def mocked_mlflow(monkeypatch):
    class MockedStartRun:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            pass

    def mocked_log_things(*args, **kwargs):
        pass

    def mocked_infer_signature(*args, **kwargs):
        return 1

    monkeypatch.setattr(connectors.mlflow, "start_run", MockedStartRun)
    monkeypatch.setattr(connectors.mlflow, "log_params", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow, "log_metric", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow, "set_tag", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow.sklearn, "log_model", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow.onnx, "log_model", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow.tensorflow, "log_model", mocked_log_things)
    monkeypatch.setattr(connectors.mlflow, "set_experiment", mocked_log_things)


@pytest.fixture
def app(mocked_mlflow: None) -> Flask:
    from app import create_app, config

    config.TESTING = True
    config.DATA_DIRECTORY = mkdtemp(prefix="CHIMP_TESTING_")
    os.mkdir(os.path.join(config.DATA_DIRECTORY, "TestingDataset"))
    app = create_app(config)

    ctx = app.app_context()
    ctx.push()

    yield app

    shutil.rmtree(config.DATA_DIRECTORY)
    ctx.pop()


@pytest.fixture
def client(app) -> FlaskClient:
    return app.test_client()


@pytest.fixture
def plugin_loader(app) -> PluginLoader:
    return app.extensions["plugin_loader"]


@pytest.fixture
def worker_manager(app) -> WorkerManager:
    return app.extensions["worker_manager"]


@pytest.fixture
def plugin(app) -> str:
    plugin_code = """
from app.plugin import BasePlugin, PluginInfo


class TestingPlugin(BasePlugin):
    def __init__(self):
        self._info = PluginInfo(
            name="Testing Plugin",
            version="1.0",
            description="test description",
            datasets={},
            arguments={"arg1": {"name": "test", "type": "str", "description": "testing arg1"}}
        )
        
    def init(self) -> PluginInfo:
        return self._info
    
    def run(self, *args, **kwargs):
        print("This is a testing plugin")
    """
    plugin_path = os.path.join(app.config["PLUGIN_DIRECTORY"], "testplugin.py")
    with open(plugin_path, "w") as f:
        f.write(plugin_code)

    yield "Testing Plugin"

    os.remove(plugin_path)


@pytest.fixture
def loaded_plugin(plugin_loader: PluginLoader, plugin) -> str:
    plugin_loader.load_plugins()
    return plugin


@pytest.fixture
def connector(app: Flask) -> BaseConnector:
    return app.extensions["connector"]


@pytest.fixture(scope="session")
def sklearn_model() -> svm.SVC:
    model = svm.SVC()
    model.fit(np.array([[0, 1], [1, 0]]), np.array([1, 0]))
    return model
