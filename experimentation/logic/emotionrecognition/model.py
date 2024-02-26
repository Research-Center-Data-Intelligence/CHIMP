"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Generating models
* Validating models"""

from typing import Union
from collections import Counter
from os import path

from logic.model import ModelGeneratorABC
from logic.emotionrecognition.__utilities import save_data_object, split_data

import pandas as pd
from numpy.random import RandomState

from talos import Scan
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    MaxPooling2D,
    SpatialDropout2D,
)
from tensorflow.keras.models import model_from_json, load_model as load_keras_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_addons.metrics import F1Score

import tf2onnx

from mlflow import start_run, log_metric, log_param, search_runs
from mlflow.models import get_model_info
from mlflow.artifacts import download_artifacts
from mlflow.onnx import log_model, load_model


class EmotionModelGenerator(ModelGeneratorABC):
    """
    A custom model generation unit for emotion recognition models.

    ...

    Implements the three parent methods '__init__()', '_generate()', and '_validate()' to generate a convolutional
    neural network (cnn) using Talos as an AutoML framework.


    Attributes
    ----------
    train_data:
        used to store the training data used for generating a cnn during Talos' hyperparameter tuning.
    validation_data:
        used to store the validation data used for validating the cnn model training.
    _curr_learning_rate:
        the base learning rate for the cnn training. If not overwritten by Talos, the default will be set to 0.05.
    _curr_optimiser:
        the optimiser used to train the cnn. If not overwritten by Talos, the default optimiser will be the Adam
        optimiser.
    _curr_convolutional_layers:
        the convolutional network architecture of the cnn with which to train. If not overwritten by Talos, a proven
        default architecture is used.
    _curr_dense_layers:
        the dense network architecture of the cnn with which to train. If not overwritten by Talos, a proven default
        architecture is used.

    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model generator to the needs of the pipeline.

    Methods
    -------
    __init__()
        Initialise according to parent interface and split the input data into deterministic sets of training and
        validation data.
    _generate() -> list[tuple]
        Either generate a single cnn model if not using Talos, function as a generator method for Talos to produce
        models for hyperparameter optimisation.
    _validate() -> list[Scan]
        Execute hyperparameter optimisation and validate each model produced by Talos' AutoML.
    """

    # Define fields with default values
    train_data = None
    validation_data = None

    _curr_learning_rate = 0.05
    _curr_optimiser = "Adam"
    _curr_convolutional_layers = [
        {
            "filters": 64,
            "kernel": (3, 3),
            "padding": "same",
            "max_pooling": (2, 2),
            "activation": "relu",
            "dropout": 0.25,
        },
        {
            "filters": 128,
            "kernel": (5, 5),
            "padding": "same",
            "max_pooling": (2, 2),
            "activation": "relu",
            "dropout": 0.25,
        },
        {
            "filters": 512,
            "kernel": (3, 3),
            "padding": "same",
            "max_pooling": (2, 2),
            "activation": "relu",
            "dropout": 0.25,
        },
        {
            "filters": 512,
            "kernel": (3, 3),
            "padding": "same",
            "max_pooling": (2, 2),
            "activation": "relu",
            "dropout": 0.25,
        },
    ]
    _curr_dense_layers = [
        {"nodes": 256, "activation": "relu", "dropout": 0.25},
        {"nodes": 512, "activation": "relu", "dropout": 0.25},
    ]

    def __init__(self, config: dict, data: Union[pd.DataFrame, any]):
        """
        Initialise according to parent interface and split the input data into deterministic sets of training and
        validation data.

        :param config: The configuration settings for the pipeline.
        :param data: The processed data features to be used for model generation.
        """
        super(EmotionModelGenerator, self).__init__(config, data)

        # Split data into train and test set, then into train and validation set
        np_random = RandomState(self._config["random_seed"])

        data_train, _ = split_data(
            self.data, self._config["train_test_fraction"], random_state=np_random
        )
        self.train_data, self.validation_data = split_data(
            data_train,
            self._config["train_validation_fraction"],
            random_state=np_random,
        )

    def _generate(self):
        """
        Either generate a single cnn model if not using Talos, function as a generator method for Talos to produce
        models for hyperparameter optimisation.

        :return: Returns the training history and resulting model.
        """

        # Only run if configured to run via talos. Can be extended to also return a model without talos, has been
        #   ignored for current implementation.
        if self._config["use_talos_automl"] and not self._config.get(
            "is_invoked_by_talos", False
        ):
            return []

        # Build sequential model
        model = Sequential()

        # Define convolutional network layers
        for conv_layer in self._curr_convolutional_layers:
            if self._curr_convolutional_layers.index(conv_layer) == 0:
                model.add(
                    Conv2D(
                        conv_layer["filters"],
                        conv_layer["kernel"],
                        padding=conv_layer["padding"],
                        input_shape=(
                            self._config["image_height"],
                            self._config["image_width"],
                            1,
                        ),
                    )
                )
            else:
                model.add(
                    Conv2D(
                        conv_layer["filters"],
                        conv_layer["kernel"],
                        padding=conv_layer["padding"],
                    )
                )

            model.add(BatchNormalization())
            model.add(Activation(conv_layer["activation"]))
            model.add(MaxPooling2D(pool_size=conv_layer["max_pooling"]))
            model.add(Dropout(conv_layer["dropout"]))

        # Flattening convolutional output for use in the dense network
        model.add(Flatten())

        # Define dense network layers
        for dense_layer in self._curr_dense_layers:
            model.add(Dense(dense_layer["nodes"]))
            model.add(BatchNormalization())
            model.add(Activation(dense_layer["activation"]))
            model.add(Dropout(dense_layer["dropout"]))

        # Define output using softmax for classification, with amount of nodes equal to amount of labels
        model.add(Dense(len(self._config["categories"]), activation="softmax"))

        # Compile model, optimise using validation loss as the main target metric, and a secondary target of loss
        learning_rate = self._curr_learning_rate
        optimiser = (
            Adam(learning_rate=learning_rate)
            if self._curr_optimiser.lower() == "adam"
            else SGD(learning_rate=learning_rate)
            if self._curr_optimiser.lower() == "sgd"
            else Adam(learning_rate=learning_rate)
        )
        model.compile(
            optimizer=optimiser,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", F1Score(len(self._config["categories"]), "micro")],
        )

        # Define class weights for model training to account for under-sampled classes
        total_sample = len(self.train_data["class_"])
        class_weights = {
            key: value / total_sample
            for key, value in Counter(self.train_data["class_"]).items()
        }

        # Define model training procedure
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto"
            ),
            EarlyStopping(
                monitor=self._config["early_stopping"]["metric"],
                min_delta=self._config["early_stopping"]["min_delta"],
                patience=self._config["early_stopping"]["patience"],
                mode=self._config["early_stopping"]["mode"],
            ),
        ]

        # DISCLAIMER: Model is non-deterministic, so results will be similar, but never exactly the same.
        history = model.fit(
            epochs=self._config["epochs"],
            x=self.train_data["image_data"],
            y=self.train_data["class_"],
            class_weight=class_weights,
            batch_size=128,
            shuffle=True,
            validation_data=(
                self.validation_data["image_data"],
                self.validation_data["class_"],
            ),
            callbacks=callbacks,
        )

        if self._config["use_talos_automl"]:
            return history, model
        else:
            return ((model, history),)

    def _validate(self, **kwargs):
        """
        Execute hyperparameter optimisation and validate each model produced by Talos' AutoML.

        :return: Returns the resulting scan object produced by Talos that contains the results for each produced model,
        and its associated network architecture.
        """

        # Generate models for the talos scan
        def generate_models(x_train, y_train, x_val, y_val, params):
            # Set up the parameters
            self._curr_learning_rate = params["learning_rate"]
            self._curr_optimiser = params["optimiser"]
            self._curr_convolutional_layers = []
            self._curr_dense_layers = []

            for _ in range(params["convolutional_layer_count"]):
                self._curr_convolutional_layers.append(
                    {
                        "filters": params["conv_filter"],
                        "kernel": params["conv_kernel_size"],
                        "padding": params["conv_padding"],
                        "max_pooling": params["conv_max_pooling"],
                        "activation": params["conv_activation"],
                        "dropout": params["conv_dropout"],
                    }
                )

            for _ in range(params["dense_layer_count"]):
                self._curr_dense_layers.append(
                    {
                        "nodes": params["dense_nodes"],
                        "activation": params["dense_activation"],
                        "dropout": params["dense_dropout"],
                    }
                )

            # Generate the resulting model
            return self._generate()

        self._config["is_invoked_by_talos"] = True

        # Scan for model options using Talos automl with a random search optimisation
        # TODO: Try and use the reduction optimiser instead of random (implement after entire project is finished.)
        scan_object = Scan(
            x=self.train_data["image_data"],
            y=self.train_data["class_"],
            params=self._config["model_parameter_optimisation"],
            model=generate_models,
            experiment_name=self._config["experiment_name"],
            x_val=self.validation_data["image_data"],
            y_val=self.validation_data["class_"],
            seed=self._config["random_seed"],
            random_method=self._config["random_method"],
            fraction_limit=self._config["random_method_fraction"],
        )

        self._config["is_invoked_by_talos"] = False

        # Return the object with all the model details
        return (scan_object,)


class MLFlowEmotionModelGenerator(EmotionModelGenerator):
    """
    A custom model generation unit for saving emotion recognition models.

    ...

    Implements the '__init__()' and _validate() methods over the base emotion recognition model generator, as to save
    the resulting models into the MLFlow artifact server.

    Methods
    -------
    __init__()
        Saves the split training and validation data to the MLFlow artifact server to distinguish between log the
        difference between data used for training, and data used for validation.
    _generate() -> list[tuple]

    _validate() -> list[Scan]
        Save each executed hyperparameter tuning step to the MLFlow tracking server. Logs parameters, resulting metrics,
        and the model itself.
    """

    def __init__(self, config: dict, data: Union[pd.DataFrame, any]):
        """
        Saves the split training and validation data to the MLFlow artifact server to distinguish between log the
        difference between data used for training, and data used for validation.

        :param config: The configuration settings for the pipeline.
        :param data: The processed data features to be used for model generation.
        """

        super(MLFlowEmotionModelGenerator, self).__init__(config, data)

        # Record training and validation data in csv format for parent run
        save_data_object(self.train_data, artifact_path="data/training")
        save_data_object(self.validation_data, artifact_path="data/validation")

    def _generate(self):
        # Intended to log in _generate(). Cannot replace logging in _validate() with logging in _generate():
        #   Information on parameters like network architecture is provided by Talos, but not directly readable from the
        #   training history or model itself without analysing the model using code.
        return super(MLFlowEmotionModelGenerator, self)._generate()

    def _validate(self):
        """
        Save each executed hyperparameter tuning step to the MLFlow tracking server. Logs parameters, resulting metrics,
        and the model itself.

        :return: Returns the resulting scan object produced by Talos that contains the results for each produced model,
        and its associated network architecture.
        """

        result = super(MLFlowEmotionModelGenerator, self)._validate()
        scan_object = result[0]

        # For each model record a child run
        mlflow_config = self._config["mlflow_config"]
        run_name_base = f"v{mlflow_config['base_model_version']}.{mlflow_config['sub_model_version']}."

        for index, model_info in scan_object.data.iterrows():
            with start_run(run_name=run_name_base + str(index + 1), nested=True) as run:
                # Record parameters
                log_param("epochs", model_info["round_epochs"])
                log_param("learning_rate", model_info["learning_rate"])
                log_param("optimiser", model_info["optimiser"])
                log_param(
                    "convolutional_layer_count", model_info["convolutional_layer_count"]
                )
                log_param("convolutional_layer_filter", model_info["conv_filter"])
                log_param(
                    "convolutional_layer_kernel_size", model_info["conv_kernel_size"]
                )
                log_param("convolutional_layer_padding", model_info["conv_padding"])
                log_param(
                    "convolutional_layer_max_pooling", model_info["conv_max_pooling"]
                )
                log_param(
                    "convolutional_layer_activation", model_info["conv_activation"]
                )
                log_param("convolutional_layer_dropout", model_info["conv_dropout"])
                log_param("dense_layer_count", model_info["dense_layer_count"])
                log_param("dense_layer_nodes", model_info["dense_nodes"])
                log_param("dense_layer_activation", model_info["dense_activation"])
                log_param("dense_layer_dropout", model_info["dense_dropout"])

                # Record metrics
                log_metric("duration", model_info["duration"])
                log_metric("loss", model_info["loss"])
                log_metric("accuracy", model_info["accuracy"])
                log_metric("f1_score", model_info["f1_score"])
                log_metric("val_loss", model_info["val_loss"])
                log_metric("val_accuracy", model_info["val_accuracy"])
                log_metric("val_f1_score", model_info["val_f1_score"])

                # Instantiate model from json and weights
                model = model_from_json(scan_object.saved_models[index])
                model.set_weights(scan_object.saved_weights[index])

                learning_rate = model_info["learning_rate"]
                optimiser = (
                    Adam(learning_rate=learning_rate)
                    if model_info["optimiser"].lower() == "adam"
                    else SGD(learning_rate=learning_rate)
                    if model_info["optimiser"].lower() == "sgd"
                    else Adam(learning_rate=learning_rate)
                )
                model.compile(
                    optimizer=optimiser,
                    loss="sparse_categorical_crossentropy",
                    metrics=[
                        "accuracy",
                        F1Score(len(self._config["categories"]), "micro"),
                    ],
                )

                #   Transform model to onnx and record model
                input_sig = [
                    tf.TensorSpec(
                        [
                            None,
                            self._config["image_height"],
                            self._config["image_width"],
                            1,
                        ],
                        tf.float32,
                    )
                ]
                onnx_model, _ = tf2onnx.convert.from_keras(model, input_sig, opset=13)

                log_model(
                    onnx_model=onnx_model,
                    artifact_path="model",
                    registered_model_name=self._config["model_name"],
                )

        return result


class EmotionCalibrationModelGenerator(ModelGeneratorABC):
    """
    A custom model generation unit for emotion recognition models for model calibration.

    ...

    Implements the three parent methods '__init__()', '_generate()', and '_validate()' to calibrate a convolutional
    neural network (cnn) already stored in the mlflow tracking and model server.


    Attributes
    ----------
    train_data:
        used to store the training data used for generating a cnn during Talos' hyperparameter tuning.
    validation_data:
        used to store the validation data used for validating the cnn model training.

    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model generator to the needs of the pipeline.

    Methods
    -------
    __init__()
        Initialise according to parent interface and split the input data into deterministic sets of training and
        validation data.
    _generate() -> list[tuple]
        Either generate a single cnn model if not using Talos, function as a generator method for Talos to produce
        models for hyperparameter optimisation.
    _validate() -> list[Scan]
        Execute hyperparameter optimisation and validate each model produced by Talos' AutoML.
    """

    # Define fields with default values
    train_data = None
    validation_data = None

    def __init__(self, config: dict, data: Union[pd.DataFrame, any]):
        """
        Initialise according to parent interface and split the input data into deterministic sets of training and
        validation data.

        :param config: The configuration settings for the pipeline.
        :param data: The processed data features to be used for model generation.
        """
        super(EmotionCalibrationModelGenerator, self).__init__(config, data)

        # Set the training and validation data partitions
        def get_data_partition(dataset, start, stop):
            return {
                "image_data": dataset["image_data"][start:stop],
                "class_": dataset["class_"][start:stop],
                "category": dataset["category"][start:stop],
            }

        partition_splits = self._config["calibration_data_entries"]
        self.train_data = get_data_partition(
            data, partition_splits[0], partition_splits[1]
        )
        self.validation_data = get_data_partition(
            data, partition_splits[1], partition_splits[2]
        )

    def _generate(self):
        """
        Calibrate a single cnn model retrieved from the mlflow production environment.

        :return: Returns the training history and resulting model.
        """

        # Get current model from production environment
        model_uri = f'models:/{self._config["model_name"]}/Production'
        production_model_info = get_model_info(model_uri)

        # Get tensorflow model and run  information via artifacts
        production_model_run = search_runs(
            experiment_names=[self._config["experiment_name"]],
            filter_string=f"run_id = '{production_model_info.run_id}'",
        )
        tensorflow_path = download_artifacts(
            run_id=production_model_info.run_id,
            artifact_path="tensorflow",
            dst_path=path.join(self._config["data_directory"]),
        )

        # Load tensorflow model and fit new data to model
        tf_model = load_keras_model(tensorflow_path)

        # Define class weights for model calibration to account for under-sampled classes
        total_sample = len(self.train_data["class_"])
        class_weights = {
            key: value / total_sample
            for key, value in Counter(self.train_data["class_"]).items()
        }

        # Define model training procedure
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto"
            ),
            EarlyStopping(
                monitor=self._config["early_stopping"]["metric"],
                min_delta=self._config["early_stopping"]["min_delta"],
                patience=self._config["early_stopping"]["patience"],
                mode=self._config["early_stopping"]["mode"],
            ),
        ]

        # Fit model to the calibration data
        # DISCLAIMER: Model is non-deterministic, so results will be similar, but never exactly the same.
        history = tf_model.fit(
            epochs=self._config["epochs"],
            x=self.train_data["image_data"],
            y=self.train_data["class_"],
            class_weight=class_weights,
            batch_size=128,
            shuffle=True,
            validation_data=(
                self.validation_data["image_data"],
                self.validation_data["class_"],
            ),
            callbacks=callbacks,
        )

        # Get model parameters and save to mlflow
        log_param("parent_model_run_id", production_model_run.at[0, "run_id"])
        log_param(
            "parent_model_run_name", production_model_run.at[0, "tags.mlflow.runName"]
        )
        log_param("epochs", history.params["epochs"])
        log_param("learning_rate", history.history["lr"][-1])
        log_param("optimiser", production_model_run.at[0, "params.optimiser"])
        log_param(
            "convolutional_layer_count",
            production_model_run.at[0, "params.convolutional_layer_count"],
        )
        log_param(
            "convolutional_layer_filter",
            production_model_run.at[0, "params.convolutional_layer_filter"],
        )
        log_param(
            "convolutional_layer_kernel_size",
            production_model_run.at[0, "params.convolutional_layer_kernel_size"],
        )
        log_param(
            "convolutional_layer_padding",
            production_model_run.at[0, "params.convolutional_layer_padding"],
        )
        log_param(
            "convolutional_layer_max_pooling",
            production_model_run.at[0, "params.convolutional_layer_max_pooling"],
        )
        log_param(
            "convolutional_layer_activation",
            production_model_run.at[0, "params.convolutional_layer_activation"],
        )
        log_param(
            "convolutional_layer_dropout",
            production_model_run.at[0, "params.convolutional_layer_dropout"],
        )
        log_param(
            "dense_layer_count", production_model_run.at[0, "params.dense_layer_count"]
        )
        log_param(
            "dense_layer_nodes", production_model_run.at[0, "params.dense_layer_nodes"]
        )
        log_param(
            "dense_layer_activation",
            production_model_run.at[0, "params.dense_layer_activation"],
        )
        log_param(
            "dense_layer_dropout",
            production_model_run.at[0, "params.dense_layer_dropout"],
        )

        return ((tf_model, history),)

    def _validate(self, **kwargs):
        """
        Execute hyperparameter optimisation and validate each model produced by Talos' AutoML.

        :return: Returns model
        """

        # Get metrics from training history
        history = self.models[0][1].history

        # Save metrics to mlflow
        log_metric("loss", history["loss"][-1])
        log_metric("accuracy", history["accuracy"][-1])
        log_metric("f1_score", history["f1_score"][-1])
        log_metric("val_loss", history["val_loss"][-1])
        log_metric("val_accuracy", history["val_accuracy"][-1])
        log_metric("val_f1_score", history["val_f1_score"][-1])

        # Return the calibrated model
        model = self.models[0][0]
        return (model,)
