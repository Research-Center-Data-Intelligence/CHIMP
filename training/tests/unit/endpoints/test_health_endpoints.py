from flask.testing import FlaskClient


class TestHealthEndpoints:
    """Tests for the health endpoints."""

    def test_ping_endpoint(self, client: FlaskClient):
        """Test the ping endpoint."""
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.text == "pong"
