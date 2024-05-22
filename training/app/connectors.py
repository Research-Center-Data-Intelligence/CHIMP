import mlflow
from abc import ABC, abstractmethod
from flask import Flask
from uuid import uuid4
from typing import Dict, Optional, Union

from app.model_type import ModelType


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
    def store_model(
        self,
        experiment_name: str,
        model: any,
        model_type: Union[ModelType, str],
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, any]] = None,
        metrics: Optional[Dict[str, any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:  # pragma: no cover
        """Store a model in the tracking service.

        The connector provides options to store the model, the inference details,
        the hyperparameters, metrics, and tags. Note that some tracking services
        might not support all types data mentioned above. In case a tracking
        service does not support a type of data (e.g. a service does not have tags)
        then this field is dropped and a warning is given.

        Parameters
        ----------
        experiment_name : str
            Name for the experiment to store
        model : any
            The model object to store
        model_type : Union[ModelType, str]
            The type of the model
        model_name : Optional[str]
            The name of the model. If no model name is specified, the experiment_name is used.
        hyperparameters : Optional[Dict[str, any]]
            Any hyperparameters used to train the model
        metrics : Optional[Dict[str, any]]
            Any metrics to log
        tags : Optional[Dict[str, str]]
            Any additional tags to store in the tracking systems.
        """
        pass

    def _init_connector(self):  # pragma: no cover
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

    def _init_connector(self):
        mlflow.set_tracking_uri(self._tracking_uri)

    def store_model(
        self,
        experiment_name: str,
        model: any,
        model_type: ModelType,
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, any]] = None,
        metrics: Optional[Dict[str, any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        mlflow.set_experiment(experiment_name)
        run_name = uuid4().hex
        with mlflow.start_run(run_name=run_name):
            if not model_name:
                model_name = experiment_name

            if hyperparameters:
                mlflow.log_params(hyperparameters)

            if metrics:
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)

            if not tags:
                tags = dict()
            if type(model_type) is str:
                tags["model_type"] = model_type
                model_type = ModelType.get_model_type(model_type)
            else:
                tags["model_type"] = model_type.value
            for tag, value in tags.items():
                mlflow.set_tag(tag, value)

            if model_type == ModelType.SKLEARN:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                print(model_info)
            if model_type == ModelType.ONNX:
                model_info = mlflow.onnx.log_model(
                    onnx_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                print(model_info)
            if model_type == ModelType.TENSORFLOW:
                model_info = mlflow.tensorflow.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                print(model_info)
            if model_type == ModelType.OTHER:
                pass
        return run_name
