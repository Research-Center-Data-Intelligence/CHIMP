from abc import ABC, abstractmethod
from datetime import datetime
from flask import Flask
from mlflow import pyfunc as mlflow_pyfunc, MlflowException
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile as MlflowNoSuchFile
from typing import Union, Optional

from app.model import BaseModel, OnnxModel


class BaseConnector(ABC):
    """Base class for any Connectors. Connectors are responsible for
    managing the connection between the serving module and the back-end
    used for storing models, parameters, and metrics. This mainly
    revolves around retrieving models from the back-end and encapsulating
    these models in Model (app.model.Model) objects that can be used
    in the rest of the application.
    """

    @abstractmethod
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Retrieves a model based on a given name.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve

        Returns
        -------
        A model object of type BaseModel or None if no model with the given name was found.
        """
        pass

    @abstractmethod
    def update_model(self, model: BaseModel) -> None:
        """Updates a given model.

        Parameters
        ----------
        model : BaseModel
            The model to update.
        """
        pass

    def init_app(self, app: Flask) -> None:
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension is already registered.
        """
        if "connector" in app.extensions:
            raise RuntimeError(
                "A 'Connector' instance has already been registered on this Flask app."
            )
        app.extensions["connector"] = self


class MLFlowConnector(BaseConnector):

    @staticmethod
    def _get_calibrated_model(model_id: str) -> any:
        model = mlflow_pyfunc.load_model(f"runs:/{model_id}/model")
        # TODO: check the type of model, then return an object of the proper Model subclass.
        return OnnxModel(model_id, {"staging": model, "production": model})

    @staticmethod
    def _get_global_model(model_name: str) -> any:
        staging = mlflow_pyfunc.load_model(f"models:/{model_name}/staging")
        production = mlflow_pyfunc.load_model(f"models:/{model_name}/production")
        # TODO: check the type of model, then return an object of the proper Model subclass.
        return OnnxModel(model_name, {"staging": staging, "production": production})

    def get_model(self, model_name: str, model_id: Optional[str] = "") -> Union[BaseModel, None]:
        try:
            if model_id:
                return self._get_calibrated_model(model_id)
            else:
                return self._get_global_model(model_name)
        except MlflowException:
            try:
                return self._get_global_model(model_name)
            except MlflowException:
                return None

    def update_model(self, model: BaseModel) -> None:
        model.updated = datetime.utcnow()
        for model_tag in model.get_model_tags():
            try:
                if model_tag in ("production", "staging"):
                    new_model = mlflow_pyfunc.load_model(f"models:/{model.name}/{model_tag}")
                else:
                    new_model = mlflow_pyfunc.load_model(f"runs:/{model.name}/model")
                model.update_model(model_tag, new_model)
            except MlflowException:
                # TODO: Add log message that model can not be updated.
                pass
            except MlflowNoSuchFile:
                # TODO: Add log message that model can not be updated.
                pass
