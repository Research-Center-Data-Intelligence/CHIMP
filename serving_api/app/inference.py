from __future__ import annotations
from datetime import datetime, timedelta
from flask import Flask
from typing import Dict, Optional

from app.connector import BaseConnector
from app.errors import ModelNotFoundError


class InferenceManager:
    """Integrates the connection with MLFlow for inference with Flask. The implementation
    of this class is inspired by the Flask-SQLAlchemy plugin, specifically the
    SQLAlchemy class.
    """

    _app: Flask = None
    _models = {}
    _connector: BaseConnector

    def __init__(self):
        pass

    def init_app(self, app: Flask, connector: BaseConnector) -> None:
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.
        connector : BaseConnector
            An Connector instance that is used to connect to the backend for model storage.

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension is already registered.
        """
        if "inference_manager" in app.extensions:
            raise RuntimeError(
                "A 'InferenceManager' instance has already been registered on this Flask app."
            )
        app.extensions["inference_manager"] = self
        self._app = app
        self._connector = connector

    def infer(
        self,
        model_name: str,
        data: Dict,
        stage: Optional[str] = "production",
        model_id: Optional[str] = "",
    ) -> Dict:
        """Infer from a model with the given name.

        Before running inference for a given model, it checks if the model is
        available. If a model is not (yet) available, it will try to load it from
        the connector. It will also check if a given model needs to be updated.

        Parameters
        ----------
        model_name : str
            The name of the model to use for inference
        data : dict
            The data to run inference on
        stage : Optional[str]
            Whether to use the production or staging model (defaults to production)
        model_id : Optional[str]
            ID for a calibrated model

        Returns
        -------
        The prediction results for the given data provided by the given model.
        """
        # Check if model is already available
        if model_name not in self._models:
            if not self._get_model(model_name, model_id):
                raise ModelNotFoundError()

        # Check if the model should check for updates
        if datetime.utcnow() - self._models[model_name].updated > timedelta(
            seconds=self._app.config["MODEL_UPDATE_INTERVAL_SECONDS"]
        ):
            self._connector.update_model(self._models[model_name])

        # Return inference based on the data
        return self._models[model_name].predict(data, stage, model_id)

    def _get_model(self, model_name: str, model_id: Optional[str] = "") -> bool:
        """Helper method to see if a model with a given name is available an to retrieve
        the model if needed.

        Parameters
        ----------
        model_name : str
            The model to search for.
        model_id : str
            A specific model ID for a calibrated model.

        Returns
        -------
        True when a model with the given name is found, False if not.
        """
        model = self._connector.get_model(model_name)
        if not model:
            return False
        self._models[model_name] = model
        return True
