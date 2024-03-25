import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app, config


@pytest.fixture
def app() -> Flask:
    config.TESTING = True
    app = create_app(config)

    ctx = app.app_context()
    ctx.push()

    yield app

    ctx.pop()


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()
