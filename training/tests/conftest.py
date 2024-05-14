import os
import pytest
import shutil
from flask import Flask
from flask.testing import FlaskClient
from tempfile import mkdtemp

from app.plugin import BasePlugin, PluginLoader
from app.worker import WorkerManager


@pytest.fixture
def app() -> Flask:
    from app import create_app, config

    config.TESTING = True
    config.DATA_DIRECTORY = mkdtemp(prefix="CHIMP_TESTING_")
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
        self._info = PluginInfo(name="Testing Plugin", version="1.0")
        
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
