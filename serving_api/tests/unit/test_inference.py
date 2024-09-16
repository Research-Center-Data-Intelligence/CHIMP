import pytest

from app.errors import ModelNotFoundError
from app.inference import InferenceManager


class TestInferenceManager:
    """Tests for the InferenceManager class."""

    def test_get_model_list(
        self,
        inference_manager: InferenceManager,
        global_model_id: str,
        global_model_name: str,
        calibrated_model_id: str,
        calibrated_model_name: str,
    ):
        """Test the get_model_list method."""
        result = inference_manager.get_models_list()
        assert global_model_id in result["available_models"]
        assert global_model_name in result["available_models"]
        assert calibrated_model_id in result["available_models"]
        assert calibrated_model_name in result["available_models"]

    def test_get_model(
        self,
        inference_manager: InferenceManager,
        global_model_id: str,
        global_model_name: str,
        calibrated_model_id: str,
        calibrated_model_name: str,
    ):
        """Test the _get_model method."""
        assert not inference_manager._get_model("does_not_exist")
        assert inference_manager._get_model(global_model_name, global_model_id)
        assert {global_model_name} == set(
            inference_manager.get_models_list()["loaded_models"]
        )
        assert inference_manager._get_model(global_model_name, calibrated_model_name)
        assert (
            calibrated_model_name
            in inference_manager.get_models_list()["loaded_models"]
        )

    def test_update_model(self, inference_manager: InferenceManager):
        """Test for the update_model method."""
        inference_manager.update_models(force=True, load_models=True)

    def test_infer(
        self,
        inference_manager: InferenceManager,
        global_model_name: str,
        calibrated_model_name: str,
    ):
        """Test for the infer method."""
        result = inference_manager.infer(global_model_name, [1, 2, 3])[0]
        assert "dense_3" in result
        assert type(result["dense_3"]) is list
        assert type(result["dense_3"][0]) is float
        result = inference_manager.infer(
            global_model_name, [1, 2, 3], model_id=global_model_name
        )[0]
        assert "dense_3" in result

        with pytest.raises(ModelNotFoundError):
            inference_manager.infer("does-not-exist", [1, 2, 3])
