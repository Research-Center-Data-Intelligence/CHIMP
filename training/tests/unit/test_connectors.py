import pytest
from flask import Flask
from sklearn.svm import SVC

from app import connectors
from app.connectors import BaseConnector, MLFlowConnector
from app.model_type import ModelType


class TestModelType:
    """Tests for the ModelType enum class."""

    def test_get_model_type(self):
        """Test the get model type method."""
        assert ModelType.get_model_type("sklearn") == ModelType.SKLEARN
        assert ModelType.get_model_type("doesnotexist") == ModelType.OTHER


class TestMlflowConnector:
    """Tests for the MlflowConnector class."""

    def test_dubbel_init_raises_runtime_error(self, app: Flask):
        """Test that an MLFlowConnector raises a RunTime error if the
        connector is already initialized."""
        conn = MLFlowConnector()
        with pytest.raises(RuntimeError):
            conn.init_app(app, "")

    def test_store_model(
        self,
        mocked_mlflow: None,
        connector: BaseConnector,
        sklearn_model: SVC,
        monkeypatch,
    ):
        """Test the store_model method of the connector."""

        class MockUuid4:
            hex = "TestHex"

        monkeypatch.setattr(connectors, "uuid4", MockUuid4)

        result = connector.store_model(
            "TestExperiment",
            "TestRun",
            sklearn_model,
            ModelType.SKLEARN,
            hyperparameters={"param1": "value1", "param2": 5},
            metrics={"accuracy": 0},
            tags={"tag1": "value"},
        )
        assert type(result) is str
        connector.store_model(
            "TestExperiment",
            "TestRun2",
            sklearn_model,
            ModelType.ONNX,
            hyperparameters={"param1": "value1", "param2": 5},
            metrics={"accuracy": 0},
        )
        connector.store_model(
            "TestExperiment",
            "TestRun3",
            sklearn_model,
            ModelType.TENSORFLOW,
            hyperparameters={"param1": "value1", "param2": 5},
            metrics={"accuracy": 0},
        )
        connector.store_model(
            "TestExperiment",
            "TestRun4",
            sklearn_model,
            ModelType.OTHER,
            hyperparameters={"param1": "value1", "param2": 5},
            metrics={"accuracy": 0},
        )
        connector.store_model(
            "TestExperiment",
            "TestRun5",
            sklearn_model,
            "onnx",
            hyperparameters={"param1": "value1", "param2": 5},
            metrics={"accuracy": 0},
        )
