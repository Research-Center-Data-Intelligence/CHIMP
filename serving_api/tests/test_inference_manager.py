import pytest
from flask import Flask

from app.inference import InferenceManager, ModelNotFoundError


def test_inference_manager_infer(app: Flask):
    """Test the infer method from the InferenceManager."""
    inference_manager = app.extensions["inference_manager"]
    assert isinstance(inference_manager, InferenceManager)
    with pytest.raises(ModelNotFoundError):
        inference_manager.infer("test_model", {})
