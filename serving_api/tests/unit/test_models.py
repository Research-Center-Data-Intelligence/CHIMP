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
        assert type(result) is tuple
        assert type(result[0]) is dict
        assert type(result[1]) is dict

    def test_get_model_by_tag(self, model: BaseModel):
        """Test the get_model_by_tag method."""
        assert model.get_model_by_tag("production")


class TestOnnxModel:
    """Tests for the Onnx model class."""

    def test_model_predict(self, mocked_mlflow):
        """Test the predict method of the Onnx model class."""

        class MockModelImpl:
            inputs = [[[], ["float"]], []]

        class MockModelMetaData:
            def to_dict(self):
                return {"meta": "data"}

        class MockedModel:
            _model_impl = MockModelImpl()
            metadata = MockModelMetaData()

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

        prediction = onnx_model.predict([1, 2])
        assert prediction[0] == {"a": [2], "b": [12]}
        assert onnx_model.predict([1, 2], stage="staging")[0] == {
            "a": [1],
            "b": [12],
        }
        assert onnx_model.predict([1, 2], model_id="some_id")[0] == {
            "a": [3],
            "b": [12],
        }
