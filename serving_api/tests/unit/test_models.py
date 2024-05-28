import numpy as np
import pytest
from onnxruntime.capi.onnxruntime_pybind11_state import (
    InvalidArgument as OnnxInvalidArgument,
)

from app.errors import InvalidDataFormatError
from app.model import BaseModel, OnnxModel


class TestModel:
    """Tests for the model class."""

    def test_model_predict(self, model: BaseModel, calibrated_model_id: str):
        """Test the model predict method."""
        with pytest.raises(InvalidDataFormatError):
            model.predict(1)

        result = model.predict([1, 2, 3])
        assert type(result) is dict

    def test_get_model_by_tag(self, model: BaseModel):
        """Test the get_model_by_tag method."""
        assert model.get_model_by_tag("production")


class TestOnnxModel:
    """Tests for the Onnx model class."""

    def test_model_predict(self):
        """Test the predict method of the Onnx model class."""

        class MockedModel:

            def __init__(self, a_val):
                self.a_val = a_val

            def predict(self, *args, **kwargs):
                print(args[0].dtype)
                if args[0].dtype == bool:
                    raise OnnxInvalidArgument
                return {"a": np.array([self.a_val]), "b": np.array([12])}

        onnx_model = OnnxModel(
            "OnnxTestModel",
            {
                "staging": MockedModel(1),
                "production": MockedModel(2),
                "some_id": MockedModel(3),
            },
        )

        assert onnx_model.predict([1, 2]) == {"a": [2], "b": [12]}
        assert onnx_model.predict([1, 2], stage="staging") == {"a": [1], "b": [12]}
        assert onnx_model.predict([1, 2], model_id="some_id") == {"a": [3], "b": [12]}
        with pytest.raises(InvalidDataFormatError):
            onnx_model.predict(True)
        with pytest.raises(InvalidDataFormatError):
            onnx_model.predict([True, True])
