from flask import Flask

from app import create_app, config


def test_create_app():
    """Test the creation of an Flask app instance."""
    config.TESTING = True
    app = create_app(config)
    assert app
    assert isinstance(app, Flask)
