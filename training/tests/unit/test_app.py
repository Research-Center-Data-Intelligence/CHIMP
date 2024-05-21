from flask import Flask

from app import create_app


class TestAppFactory:
    """Tests for the app factory function (`app.create_app)`."""

    def test_create_app(self, capfd):
        """Test if an app instance can be created."""
        app = create_app()
        assert type(app) is Flask
        out, err = capfd.readouterr()
        assert not err
        assert "Example Plugin" in out
        assert "Example 2 Plugin" in out
