import os
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.plugin import BasePlugin, PluginLoader


@pytest.fixture
def app() -> Flask:
    from app import create_app, config

    config.TESTING = True
    app = create_app(config)

    ctx = app.app_context()
    ctx.push()

    yield app

    ctx.pop()


@pytest.fixture
def client(app) -> FlaskClient:
    return app.test_client()


@pytest.fixture
def plugin_loader(app) -> PluginLoader:
    return app.extensions["plugin_loader"]


@pytest.fixture
def plugin(app) -> BasePlugin:
    plugin_code = """
from app.plugin import BasePlugin, PluginInfo


class TestingPlugin(BasePlugin):
    def __init__(self):
        self._info = PluginInfo(name="Testing Plugin", version="1.0")
        
    def init(self) -> PluginInfo:
        return self._info
    
    def run(self):
        print("This is a testing plugin")
    """
    plugin_path = os.path.join(app.config["PLUGIN_DIRECTORY"], "testplugin.py")
    with open(plugin_path, "w") as f:
        f.write(plugin_code)

    yield "Testing Plugin"

    os.remove(plugin_path)
