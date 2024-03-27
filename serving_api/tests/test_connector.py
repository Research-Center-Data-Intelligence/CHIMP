from flask import Flask

from app.connector import BaseConnector


def test_connector_get_model(app: Flask):
    """Test the get_model method from the Connector."""
    connector = app.extensions["connector"]
    assert issubclass(type(connector), BaseConnector)
    result = connector.get_model("test_model")
    assert result is None
