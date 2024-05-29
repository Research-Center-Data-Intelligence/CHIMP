from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List
from onnxruntime.capi.onnxruntime_pybind11_state import (
    InvalidArgument as OnnxInvalidArgument,
)

from app.errors import InvalidDataFormatError, InvalidModelIdOrStage


class BaseModel(ABC):
    """Base class for models."""

    name: str
    _models: Dict[str, Any] = {}
    updated: datetime

    def __init__(self, model_name: str, models: Dict[str, Any]):
        self.name = model_name
        self._models = models
        self.updated = datetime.utcnow()

    def get_model(
        self, stage: Optional[str] = "production", model_id: Optional[str] = ""
    ) -> Any:
        """Get a model by ID or stage name, raises a `InvalidModelIdOrStage` if
        no model with a given AI or stage exists.
        """
        model = None
        if model_id and model_id in self._models:
            model = self._models[model_id]
        elif stage in self._models:
            model = self._models[stage]

        if not model:
            raise InvalidModelIdOrStage(
                f"No model with ID '{model_id}' or with stage '{stage}' found"
            )
        return model

    @abstractmethod
    def predict(
        self,
        data: Any,
        stage: Optional[str] = "production",
        model_id: Optional[str] = "",
    ) -> Any:  # pragma: no cover
        """Run a prediction on the given data.

        Parameters
        ----------
        data : Any
            The data to run a prediction on.
        stage : Optional[str]
            The stage of the model to use (defaults to production).
        model_id : Optional[str]
            Optionally use a calibrated model.

        Returns
        -------
        The prediction for the given data.
        """
        pass

    def update_model(self, tag: str, updated_model: Any) -> None:
        """Update a model with a given tag (either staging or a calibrated model).

        Parameters
        ----------
        tag : str
            The tag for the model to update.
        updated_model : Any
            New model to store for the given tag.
        """
        self._models[tag] = updated_model
        self.updated = datetime.utcnow()

    def get_model_tags(self) -> List[str]:
        """Get a list of available tags.

        Returns
        -------
        A list of available tags for this model.
        """
        return list(self._models.keys())

    def get_model_by_tag(self, tag: str) -> Optional[Any]:
        """Get a model based on a given tag.

        Parameters
        ----------
        tag : str
            The tag to return a model for.

        Returns
        -------
        The model for the given tag or None if no model with a given tag is found
        """
        return self._models.get(tag)


class OnnxModel(BaseModel):
    """Implementation of BaseModel for ONNX models."""

    def predict(
        self,
        data: Any,
        stage: Optional[str] = "production",
        model_id: Optional[str] = "",
    ) -> Any:
        if type(data) is not list:
            raise InvalidDataFormatError("Expected a list as data input")
        try:
            model = self.get_model(stage=stage, model_id=model_id)
            data = np.asarray(data)
            prediction = model.predict(data)
            return {k: v.tolist() for k, v in prediction.items()}
        except OnnxInvalidArgument:
            raise InvalidDataFormatError()


class PyTorchModel(BaseModel):  # pragma: no cover
    """(WIP) implementation of BaseModel for PyTorch model.

    This implementation has not been tested yet. Currently, there are a number of issues
    with the import system between the training module and this module.
    """

    def predict(
        self,
        data: Any,
        stage: Optional[str] = "production",
        model_id: Optional[str] = "",
    ) -> Any:

        if type(data) is not list:
            raise InvalidDataFormatError("Expected a list as data input")
        try:
            if model_id and model_id in self._models:
                model = self._models[model_id]
            else:
                model = self._models[stage]
            data = np.asarray(data)
            prediction = model(data)
            return {k: v.tolist() for k, v in prediction.items()}
        except OnnxInvalidArgument:
            raise InvalidDataFormatError()
