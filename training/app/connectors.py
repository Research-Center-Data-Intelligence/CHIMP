import mlflow
from abc import ABC, abstractmethod
from flask import Flask
from uuid import uuid4
from typing import Dict, Optional, Union

from app.errors import RunNotFoundError, ModelNotFoundError
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
        run_name: str,
        model: any,
        model_type: Union[ModelType, str],
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, any]] = None,
        metrics: Optional[Dict[str, any]] = None,
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, any]] = None,
        datasets: Optional[Dict[str, str]] = None,
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
        run_name : str
            Name of the run
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
            Any additional tags to store in the tracking system.
        artifacts : Optional[Dict[str, str]]
            Any additional artifacts that need to be uploaded to the tracking system. The key
            is the name of the artifact, the value is the path to the artifact on the local
            system.
        datasets : Optional[Dict[str, str]]
            Any datasets that need to be uploaded to the tracking system. The key is the name of the
            dataset, the value is the path to the dataset.
        """
        pass

    @abstractmethod
    def get_artifact(
        self,
        save_to: str,
        model_name: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        artifact_path: Optional[str] = "model",
    ) -> str:  # pragma: no cover
        """Retrieve an artifact from the tracking service.

        Parameters
        ----------
        save_to : str
            The path to save the artifact to (this should be inside the tmp directory created
            for a task.
        model_name : str
            Name of the model to retrieve the artifact for.
        experiment_name : str
            Name of the experiment to retrieve the artifact for
        run_name : Optional[str]
            Name of the run to load
        artifact_path : Optional[artifact_path]

        Returns
        -------
        The model path where the model is stored.
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
        run_name: str,
        model: any,
        model_type: ModelType,
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, any]] = None,
        metrics: Optional[Dict[str, any]] = None,
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = {},
        datasets: Optional[Dict[str, str]] = {},
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

            if artifacts:
                for artifact_name, artifact_location in artifacts.items():
                    mlflow.log_artifact(artifact_location, artifact_name)

            if datasets:
                for dataset_name, dataset_location in datasets.items():
                    mlflow.log_artifact(dataset_location, f"dataset_{dataset_name}")

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
                    conda_env=None,
                    code_paths=None,
                )
                print(model_info)
            if model_type == ModelType.TENSORFLOW:
                model_info = mlflow.tensorflow.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    conda_env=None,
                    code_paths=None,
                )
                print(model_info)
            if model_type == ModelType.PYTORCH:
                model_info = mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    conda_env=None,
                    code_paths=None,
                )
            if model_type == ModelType.OTHER:
                pass
        return run_name

    def get_artifact(
        self,
        save_to: str,
        model_name: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        artifact_path: Optional[str] = "model",
    ) -> str:
        try:
            production_model_info = mlflow.models.get_model_info(
                f"models:/{model_name}/Production"
            )
        except mlflow.exceptions.MlflowException:
            raise ModelNotFoundError()
        if run_name:
            run_info = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"run_name = '{run_name}'",
            )
            if len(run_info) < 1:
                raise RunNotFoundError()
            run_id = run_info[0].run_id
        else:
            run_id = production_model_info.run_id

        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=save_to
        )
        return model_path
