import pytest
from flask import Flask
from mlflow.exceptions import MlflowException

from app.connectors import BaseConnector, MLFlowConnector
from app.model import BaseModel


class TestMlflowConnector:
    """Tests for the MlflowConnector class."""

    def test_dubble_init_raises_runtime_error(self, app: Flask):
        """Test whether trying to initialize a connector twice raises
        a RuntimeError.
        """
        conn = MLFlowConnector()
        with pytest.raises(RuntimeError):
            conn.init_app(app, "")

    def test_get_available_models(
        self,
        connector: BaseConnector,
        calibrated_model_id: str,
        calibrated_model_name: str,
        global_model_id: str,
        global_model_name: str,
    ):
        """Test the get_available_models method."""
        assert connector.get_available_models() == {
            calibrated_model_name,
            calibrated_model_id,
            global_model_id,
            global_model_name,
        }

    def test_get_calibrated_model(
        self, connector: BaseConnector, calibrated_model_name: str
    ):
        """Test the get_calibrated_model method."""
        model = connector._get_calibrated_model(calibrated_model_name)
        assert model
        assert issubclass(type(model), BaseModel)
        assert model.name == calibrated_model_name

        with pytest.raises(MlflowException):
            connector._get_calibrated_model("some_random_model_name")

    def test_get_global_model(self, connector: BaseConnector, global_model_name: str):
        """Test the get_global_model method."""
        model = connector._get_global_model(global_model_name)
        assert model
        assert issubclass(type(model), BaseModel)
        assert model.name == global_model_name

        with pytest.raises(MlflowException):
            connector._get_global_model("some_random_model_name")

    def test_get_model(
        self,
        connector: BaseConnector,
        global_model_name: str,
        calibrated_model_name: str,
    ):
        """Test the get_model method."""
        model = connector.get_model(global_model_name)
        assert model
        assert issubclass(type(model), BaseModel)
        assert model.name == global_model_name

        model = connector.get_model(global_model_name, calibrated_model_name)
        assert model
        assert issubclass(type(model), BaseModel)
        assert model.name == calibrated_model_name

        model = connector.get_model(global_model_name, "random_calibrated_model_id")
        assert model
        assert issubclass(type(model), BaseModel)
        assert model.name == global_model_name

        model = connector.get_model("some_random_model_name", "random_model_id")
        assert not model

    def test_update_model(self, connector: BaseConnector, global_model_name):
        """Test the update model method"""
        model = connector.get_model(global_model_name)
        old_update_time = model.updated
        connector.update_model(model)
        assert model.updated > old_update_time
