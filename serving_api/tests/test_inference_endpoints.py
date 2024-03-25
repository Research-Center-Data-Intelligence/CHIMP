import json
from flask.testing import FlaskClient


class TestInferFromModel:
    """Tests for the infer_from_model route (/model/<model_name>/infer)."""

    def test_infer_from_model_basic(self, client: FlaskClient):
        """Test the infer_from_model route with basic use."""
        resp = client.post(
            "/model/test/infer",
            content_type="application/json",
            data=json.dumps({"inputs": [1, 2]}),
        )
        assert resp.status_code == 200, resp.json
        assert resp.is_json
        assert resp.json == {
            "data": {"a": 0.9, "b": 0.1},
            "status": "inference from model test success",
        }
