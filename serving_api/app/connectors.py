import mlflow
from abc import ABC, abstractmethod
from datetime import datetime
from flask import Flask
from mlflow import pyfunc as mlflow_pyfunc, MlflowException, MlflowClient
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile as MlflowNoSuchFile
from typing import Union, Optional, Set

from app.model import BaseModel, OnnxModel, PyTorchModel


class BaseConnector(ABC):
    """Base class for any Connectors. Connectors are responsible for
    managing the connection between the serving module and the back-end
    used for storing models, parameters, and metrics. This mainly
    revolves around retrieving models from the back-end and encapsulating
    these models in Model (app.model.Model) objects that can be used
    in the rest of the application.
    """

    _tracking_uri: str

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

    @abstractmethod
    def get_available_models(self) -> Set:
        pass

    def _init_connector(self):
        """Helper method for any connector specific initialization."""
        pass

    def init_app(self, app: Flask, tracking_uri: str) -> None:
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.
        tracking_uri : str
            The connection URI for the tracking server

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension is already registered.
        """
        if "connector" in app.extensions:
            raise RuntimeError(
                "A 'Connector' instance has already been registered on this Flask app."
            )
        self._tracking_uri = tracking_uri
        self._init_connector()
        app.extensions["connector"] = self


class MLFlowConnector(BaseConnector):
    _client: MlflowClient

    def _init_connector(self):
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient()

    @staticmethod
    def _get_calibrated_model(model_id: str) -> any:
        runs = mlflow.search_runs(
            search_all_experiments=True, filter_string=f"run_name = {model_id}"
        )
        if len(runs) != 1:
            raise MlflowException(f"Could not find run with name {model_id}")
        run_id = runs.iloc[0].run_id
        model = mlflow_pyfunc.load_model(f"runs:/{run_id}/model")
        # TODO: check the type of model, then return an object of the proper Model subclass.
        return OnnxModel(model_id, {"staging": model, "production": model})

    @staticmethod
    def _get_global_model(model_name: str) -> any:
        staging = mlflow_pyfunc.load_model(f"models:/{model_name}/staging")
        production = mlflow_pyfunc.load_model(f"models:/{model_name}/production")
        # TODO: check the type of model, then return an object of the proper Model subclass.
        return OnnxModel(model_name, {"staging": staging, "production": production})

    def get_model(
        self, model_name: str, model_id: Optional[str] = ""
    ) -> Union[BaseModel, None]:
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
                    new_model = mlflow_pyfunc.load_model(
                        f"models:/{model.name}/{model_tag}"
                    )
                else:
                    new_model = mlflow_pyfunc.load_model(f"runs:/{model.name}/model")
                model.update_model(model_tag, new_model)
            except MlflowException:
                # TODO: Add log message that model can not be updated.
                pass
            except MlflowNoSuchFile:
                # TODO: Add log message that model can not be updated.
                pass

    def get_available_models(self) -> Set:
        available_models = set()

        models = self._client.search_model_versions("")
        for model in models:
            available_models.add(model.name)
            available_models.add(model.run_id)
            run_info = self._client.get_run(model.run_id)
            available_models.add(run_info.info.run_name)

        return available_models
