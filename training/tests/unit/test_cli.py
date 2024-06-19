import pytest
from flask import Flask
from flask.testing import FlaskCliRunner

from app.cli import ping


@pytest.fixture
def cli(app: Flask) -> FlaskCliRunner:
    return app.test_cli_runner()


class TestCli:
    """Tests for the CLI."""

    def test_ping(self, cli: FlaskCliRunner):
        """Test the ping command."""
        result = cli.invoke(ping)
        assert result.exit_code == 0
