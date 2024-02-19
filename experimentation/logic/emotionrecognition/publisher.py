"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Testing models
* Publishing models"""

from logic.publisher import ModelPublisherABC
from logic.emotionrecognition.__utilities import save_data_object, split_data

from os import path
import heapq
from tempfile import TemporaryDirectory

from numpy.random import RandomState

import tensorflow as tf
from tensorflow.keras.models import model_from_json, save_model as tf_save_model, load_model as tf_load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.metrics import F1Score

import onnx
import tf2onnx
from mlflow import MlflowClient, start_run, log_param, log_metric, log_artifacts
from mlflow.onnx import log_model
from mlflow.models import get_model_info


class EmotionModelPublisher(ModelPublisherABC):
    """
    A custom model publisher unit for emotion recognition models.

    ...

    Implements the three parent methods '_test_models()' and '_publish_models()' to publish valid convolutional neural
    networks (cnn) using Talos as an AutoML framework.

    Attributes
    ----------
    test_data:
        used to store the test data used for testing the best cnn-s during Talos' hyperparameter tuning.

    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model publisher to the needs of the pipeline.

    Methods
    -------
    _test_models() -> list[]
        Get n-best models according model generator and test those models using the remainder of the unseen data.
    _publish_models() -> list[]
        Publish n-best models based on the evaluation test of the selected models. Models will be saved as an
        onnx-model.
    """

    test_data = None

    def _test_models(self):
        """
        Get n-best models according model generator and test those models using the remainder of the unseen data.

        :return: Returns the results of those models and their corresponding results.
        """

        # Split data into train and test set using the same set as used in the model component, discard training
        _, self.test_data = split_data(self._data, self._config['train_test_fraction'],
                                       random_state=RandomState(self._config['random_seed']))

        # Get best n models according to training and validation results
        talos_scan = self.models[0]
        scan_results = zip(talos_scan.saved_models, talos_scan.saved_weights, talos_scan.data.to_dict('records'))

        n_value = self._config['best_model_test_count']
        evaluation_metric = self._config['best_model_test_metric']

        best_n_function = heapq.nlargest if evaluation_metric != 'loss' and evaluation_metric != 'val_loss' \
            else heapq.nsmallest
        holdout_best_models = best_n_function(n_value, scan_results,
                                              key=lambda model_results: model_results[2][evaluation_metric])

        # Evaluate the n best models according to the scan object
        models = []

        for model_json, weights, data in holdout_best_models:
            model = model_from_json(model_json)
            model.set_weights(weights)

            learning_rate = data['learning_rate']
            optimiser = Adam(learning_rate=learning_rate) if data['optimiser'].lower() == 'adam' else \
                SGD(learning_rate=learning_rate) if data['optimiser'].lower() == 'sgd' else \
                Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
                          metrics=['accuracy', F1Score(len(self._config['categories']), 'micro')])

            loss, acc, f1_score = model.evaluate(self.test_data['image_data'], self.test_data['class_'])
            models.append({
                'model': model,
                'data': data,
                'loss': loss,
                'accuracy': acc,
                'f1_score': f1_score,
            })

        return models

    def _publish_models(self):
        """
        Publish n-best models based on the evaluation test of the selected models. Models will be saved as an
        onnx-model.

        :return: Returns the published/saved models.
        """

        # Get best n models based on the evaluation during the test phase
        n_value = self._config['best_model_publish_count']
        evaluation_metric = self._config['best_model_publish_metric']

        best_n_function = heapq.nlargest if evaluation_metric != 'loss' and evaluation_metric != 'val_loss' \
            else heapq.nsmallest
        best_models = best_n_function(n_value, self.models,
                                      key=lambda model: model[evaluation_metric])

        # Save all best models
        if path.exists(self._config["publish_directory"]):
            input_sig = [tf.TensorSpec([None, self._config['image_height'], self._config['image_width'], 1], tf.float32)]

            for i in range(len(best_models)):
                onnx_model, _ = tf2onnx.convert.from_keras(best_models[i]['model'], input_sig, opset=13)
                onnx.save(onnx_model, f'{self._config["publish_directory"]}/model-{i+1}.onnx')

        return best_models


class MLFlowEmotionModelPublisher(EmotionModelPublisher):
    """
    A custom model publisher unit for saving emotion recognition models.

    ...

    Implements the '_test_models()' and _publish_models() methods over the base emotion recognition model generator, as
    to save and publish the selected models into the MLFlow artifact server.

    Methods
    -------
    _test_models() -> list[tuple]
        Save the model tests and its results to the MLFlow server.
    _publish_models() -> list[Scan]
        Upload the best-tested model in the MLFlow tracking server as the MLFlow main run's results, and put it into
        staging.
    """

    def _test_models(self):
        """
        Save the model tests and its results to the MLFlow server.

        :return: Returns the results of the tests and their associated models.
        """

        models = super(MLFlowEmotionModelPublisher, self)._test_models()

        # Record test data in npy format for parent run
        save_data_object(self.test_data, artifact_path='data/test')

        # For each evaluation record a child run
        mlflow_config = self._config['mlflow_config']
        run_name_base = f"v{mlflow_config['base_model_version']}.{mlflow_config['sub_model_version']}."

        for index, model_entry in enumerate(models):
            model = model_entry['model']
            model_info = model_entry['data']

            with start_run(run_name=run_name_base + str(index + 1) + ' (eval)', nested=True) as run:
                # Record parameters
                log_param('epochs', model_info['round_epochs'])
                log_param('learning_rate', model_info['learning_rate'])
                log_param('optimiser', model_info['optimiser'])
                log_param('convolutional_layer_count', model_info['convolutional_layer_count'])
                log_param('convolutional_layer_filter', model_info['conv_filter'])
                log_param('convolutional_layer_kernel_size', model_info['conv_kernel_size'])
                log_param('convolutional_layer_padding', model_info['conv_padding'])
                log_param('convolutional_layer_max_pooling', model_info['conv_max_pooling'])
                log_param('convolutional_layer_activation', model_info['conv_activation'])
                log_param('convolutional_layer_dropout', model_info['conv_dropout'])
                log_param('dense_layer_count', model_info['dense_layer_count'])
                log_param('dense_layer_nodes', model_info['dense_nodes'])
                log_param('dense_layer_activation', model_info['dense_activation'])
                log_param('dense_layer_dropout', model_info['dense_dropout'])

                # Record metrics
                log_metric('duration', model_info['duration'])
                log_metric('loss', model_entry['loss'])
                log_metric('test_accuracy', model_entry['accuracy'])
                log_metric('f1_score', model_entry['f1_score'])

                #   Transform model to onnx and record model
                input_sig = [
                    tf.TensorSpec([None, self._config['image_height'], self._config['image_width'], 1], tf.float32)]
                onnx_model, _ = tf2onnx.convert.from_keras(model, input_sig, opset=13)

                log_model(onnx_model=onnx_model, artifact_path="model",
                          registered_model_name=self._config['model_name'])

        return models

    def _publish_models(self):
        """
        Upload the best-tested model in the MLFlow tracking server as the MLFlow main run's results, and put it into
        staging.

        :return: Returns the selected best-tested models.
        """
        best_models = super(MLFlowEmotionModelPublisher, self)._publish_models()

        best_model_entry = best_models[0]
        best_model = best_model_entry['model']
        best_model_info = best_model_entry['data']

        # Record parameters for best model
        log_param('epochs', best_model_info['round_epochs'])
        log_param('learning_rate', best_model_info['learning_rate'])
        log_param('optimiser', best_model_info['optimiser'])
        log_param('convolutional_layer_count', best_model_info['convolutional_layer_count'])
        log_param('convolutional_layer_filter', best_model_info['conv_filter'])
        log_param('convolutional_layer_kernel_size', best_model_info['conv_kernel_size'])
        log_param('convolutional_layer_padding', best_model_info['conv_padding'])
        log_param('convolutional_layer_max_pooling', best_model_info['conv_max_pooling'])
        log_param('convolutional_layer_activation', best_model_info['conv_activation'])
        log_param('convolutional_layer_dropout', best_model_info['conv_dropout'])
        log_param('dense_layer_count', best_model_info['dense_layer_count'])
        log_param('dense_layer_nodes', best_model_info['dense_nodes'])
        log_param('dense_layer_activation', best_model_info['dense_activation'])
        log_param('dense_layer_dropout', best_model_info['dense_dropout'])

        # Record metrics for best model
        log_metric('duration', best_model_info['duration'])
        log_metric('test_loss', best_model_entry['loss'])
        log_metric('test_accuracy', best_model_entry['accuracy'])
        log_metric('test_f1_score', best_model_entry['f1_score'])

        #   Transform best model to onnx and record model
        input_sig = [
            tf.TensorSpec([None, self._config['image_height'], self._config['image_width'], 1], tf.float32)]
        onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_sig, opset=13)

        new_model_info = log_model(onnx_model=onnx_model, artifact_path='model',
                                   registered_model_name=self._config['model_name'])

        # Upload best tensorflow version using temporary directory as artifact
        #   Note: Important for calibration purposes. (assets folder does not get uploaded, as it is empty)
        with TemporaryDirectory() as tmpdir:
            tf_save_model(best_model, tmpdir)
            log_artifacts(tmpdir, 'tensorflow')

        # Put best model in staging, and current model in staging into production

        # Archive production model and promote staging model to production model
        # (TODO: quality measure to only update if better??)
        client = MlflowClient()

        try:
            curr_staging_model_info = get_model_info(f'models:/{self._config["model_name"]}/Staging')
            model_run_id = curr_staging_model_info.run_id
            model_version_info = client.search_model_versions(f"run_id = '{model_run_id}'")[0]

            client.transition_model_version_stage(self._config['model_name'], model_version_info.version,
                                                  stage='production', archive_existing_versions=True)
        except Exception:
            # Generic exception handling. Bad practice, but could not find the exception thrown when client has couldn't
            #   return model data
            pass

        # Promote new model to staging model
        model_run_id = new_model_info.run_id
        model_version_info = client.search_model_versions(f"run_id = '{model_run_id}'")[0]

        client.transition_model_version_stage(self._config['model_name'], model_version_info.version,
                                              stage='staging', archive_existing_versions=False)

        return best_models


class EmotionCalibrationModelPublisher(ModelPublisherABC):
    """
    A custom model publisher unit for emotion recognition models for model calibration.

    ...

    Implements the three parent methods '_test_models()' and '_publish_models()' to publish calibrated convolutional
    neural networks (cnn) to mlflow for user specific use-cases.

    Attributes
    ----------
    test_data:
        used to store the test data used for testing the calibrated cnn.

    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model publisher to the needs of the pipeline.

    Methods
    -------
    _test_models() -> list[]
        Get n-best models according model generator and test those models using the remainder of the unseen data.
    _publish_models() -> list[]
        Publish n-best models based on the evaluation test of the selected models. Models will be saved as an
        onnx-model.
    """

    test_data = None

    def _test_models(self):
        """
        Test those models using the reserved test data.

        :return: Returns the results of the model.
        """

        # Set test data
        def get_data_partition(dataset, start, stop):
            return {
                'image_data': dataset['image_data'][start:stop],
                'class_': dataset['class_'][start:stop],
                'category': dataset['category'][start:stop]
            }

        partition_splits = self._config['calibration_data_entries']
        self.test_data = get_data_partition(self._data, partition_splits[2], partition_splits[3])

        # Evaluate model
        model = self.models[0]
        loss, acc, f1_score = model.evaluate(self.test_data['image_data'], self.test_data['class_'])

        # Log metrics to mlflow
        log_metric('test_loss', loss)
        log_metric('test_accuracy', acc)
        log_metric('test_f1_score', f1_score)

        return model,

    def _publish_models(self):
        """
        Publish n-best models based on the evaluation test of the selected models. Models will be saved as an
        onnx-model.

        :return: Returns the published/saved models.
        """

        model = self.models[0]

        # Save model to mlflow in onnx format
        # NOTE: Don't put calibrated model in production -> User specific model, not a globally acknowledge model
        input_sig = [tf.TensorSpec([None, self._config['image_height'], self._config['image_width'], 1], tf.float32)]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_sig, opset=13)
        log_model(onnx_model=onnx_model, artifact_path='model', registered_model_name=self._config['model_name'])

        # Save model to mlflow in tensorflow format using a temporary directory as an intermediate
        # NOTE: Important for calibration purposes. (assets folder does not get uploaded, as it is empty)
        with TemporaryDirectory() as tmpdir:
            tf_save_model(model, tmpdir)
            log_artifacts(tmpdir, 'tensorflow')

        return self.models
