from abc import ABC, abstractmethod
from datetime import datetime
from flask import Flask
from mlflow import pyfunc as mlflow_pyfunc, MlflowException
from typing import Union

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
    def get_model(self, model_name: str) -> Union[BaseModel, None]:
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

    def get_model(self, model_name: str) -> Union[BaseModel, None]:
        try:
            staging_model = mlflow_pyfunc.load_model(f"models:/{model_name}/staging")
            production_model = mlflow_pyfunc.load_model(
                f"models:/{model_name}/production"
            )
            return OnnxModel(
                model_name, {"staging": staging_model, "production": production_model}
            )
        except MlflowException as ex:
            return None

    def update_model(self, model: BaseModel) -> None:
        model.updated = datetime.utcnow()
        for model_name in model.get_model_tags():
            pass
