from flask.testing import FlaskClient


class TestInferenceEndpoints:
    """Test the inference endpoints."""

    def test_get_models_endpoint(
        self,
        client: FlaskClient,
        global_model_id: str,
        global_model_name: str,
        calibrated_model_id: str,
        calibrated_model_name: str,
    ):
        """Test the inference endpoint."""
        resp = client.get("/model")
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "updated_models" in data and not data["updated_models"]
        assert "status" in data and data["status"] == "successfully retrieved models"
        assert "data" in data
        assert "loaded_models" in data["data"] and data["data"]["loaded_models"] == []
        assert "available_models" in data["data"]
        assert global_model_id in data["data"]["available_models"]
        assert global_model_name in data["data"]["available_models"]
        assert calibrated_model_id in data["data"]["available_models"]
        assert calibrated_model_name in data["data"]["available_models"]

        resp = client.get("/model?reload_models=true")
        assert resp.status_code == 200

    def test_infer_from_model_endpoint(
        self,
        client: FlaskClient,
        global_model_id: str,
        global_model_name: str,
        calibrated_model_id: str,
        calibrated_model_name: str,
    ):
        """Test the infer from model endpoint."""
        resp = client.post("/model/doesnotexist/infer", json={"inputs": [1, 2, 3]})
        assert resp.status_code == 404

        resp = client.post(f"/model/{global_model_name}/infer", data="abc")
        assert resp.status_code == 400

        resp = client.post(f"/model/{global_model_name}/infer", json={"abc": 123})
        assert resp.status_code == 400

        resp = client.post(f"/model/{global_model_name}/infer", json={"inputs": 12})
        assert resp.status_code == 400

        resp = client.post(
            f"/model/{global_model_name}/infer", json={"inputs": [1, 2, 3]}
        )
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert (
            "status" in data
            and data["status"] == f"inference from model {global_model_name} success"
        )
        assert "predictions" in data and type(data["predictions"]["dense_3"]) is list

        resp = client.post(
            f"/model/{global_model_name}/infer?stage=doesnotexist",
            json={"inputs": [1, 2, 3]},
        )
        assert resp.status_code == 400
