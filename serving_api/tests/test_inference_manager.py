from flask import Flask
from app.inference import InferenceManager


def test_inference_manager_infer(app: Flask):
    """Test the infer method from the InferenceManager."""
    inference_manager = app.extensions["inference_manager"]
    assert isinstance(inference_manager, InferenceManager)
    result = inference_manager.infer("test_model", {})
    assert result == {"a": 0.9, "b": 0.1}
