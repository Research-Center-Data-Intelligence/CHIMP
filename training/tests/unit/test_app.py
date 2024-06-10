from flask import Flask

from app import create_app


class TestAppFactory:
    """Tests for the app factory function (`app.create_app)`."""

    def test_create_app(self, capfd, minio_mock):
        """Test if an app instance can be created."""
        app = create_app()
        assert type(app) is Flask
