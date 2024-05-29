from __future__ import annotations
from datetime import datetime, timedelta
from flask import Flask
from typing import Dict, Optional, Set, List

from app.connectors import BaseConnector
from app.errors import ModelNotFoundError


class InferenceManager:
    """Integrates the connection with the model back-end for inference with Flask. The implementation
    of this class is inspired by the Flask-SQLAlchemy plugin, specifically the
    SQLAlchemy class.
    """

    _app: Flask = None
    _models = {}
    _available_models: Set = (
        set()
    )  # Set is used here instead of List so that time complexity of lookups is O(1).
    _connector: BaseConnector
    _last_update: datetime

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
        self._last_update = datetime.utcnow()
        self.update_models(force=True)

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
        # Check if the model should check for updates
        self.update_models()

        # Select correct model if available, or raise an error if the model is not available.
        selected_model = None
        if model_id:  # Calibrated model selection
            if model_id in self._models:
                selected_model = self._models[model_id]
            elif model_id in self._available_models:
                if self._get_model(model_name, model_id):
                    selected_model = self._models[model_id]

        if not selected_model:  # Global model selection
            if model_name in self._models:
                selected_model = self._models[model_name]
            elif model_name in self._available_models:
                if self._get_model(model_name):
                    selected_model = self._models[model_name]

        if not selected_model:  # No model found
            raise ModelNotFoundError()

        # Return inference based on the data
        # TODO: add metadata (e.g. which model was selected) to return
        return selected_model.predict(data, stage, model_id)

    def update_models(
        self, force: Optional[bool] = False, load_models: Optional[bool] = False
    ) -> None:
        """Update the available models.

        A method for updating the models used. By default, it only updates the models if the
        time since the last update is greater than the configured interval, unless the
        force option is used. Additionally, by default, it only checks whether the models
        are available, but does not actually load them, unless load_models is used.

        Parameters
        ----------
        force : Optional[bool]
            Optional parameters specifying whether to force updates or not.
        load_models : Optional[bool]
            Optional parameter specifying whether to actually load the models,
            or only check if they are available.
        """
        time_threshold = timedelta(
            seconds=self._app.config["MODEL_UPDATE_INTERVAL_SECONDS"]
        )
        if force or datetime.utcnow() - self._last_update > time_threshold:
            self._available_models = self._connector.get_available_models()

            if load_models:
                for model in self._models.values():
                    if force or model.updated > time_threshold:
                        self._connector.update_model(model)

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
        model = self._connector.get_model(model_name, model_id)
        if not model:
            return False
        if model_id:
            self._models[model_id] = model
        else:
            self._models[model_name] = model
        return True

    def get_models_list(self) -> Dict[str, List[str]]:
        """Get a list of loaded and available models.

        Returns
        -------
        A dictionary containing two lists, one of the loaded models, another one of the
        available models.
        """
        return {
            "loaded_models": [model.name for model in self._models.values()],
            "available_models": list(self._available_models),
        }
