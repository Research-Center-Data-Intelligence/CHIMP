import os
import time
from threading import Thread

import numpy as np
from mlflow import pyfunc as mlflow_pyfunc, search_runs as mlflow_search_runs, MlflowException
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument, NoSuchFile

from serving.messaging import messaging_manager


class InferenceManager:
    do_keep_updating: bool = True   # No way to stop updating except for exiting the application. Can be implemented tho
    _update_interval: int
    _models: dict[str, mlflow_pyfunc.PyFuncModel]
    _calibrated_model_retrieval_list: list[str]
    _DEFAULT_RESPONSE: dict[str, list[tuple]] = {'node_007': [('', 0.0)]}

    def __init__(self, model_update_interval: int = 60):
        self._models = {}
        self._calibrated_model_retrieval_list = []
        # TODO: Replace threat update with update triggered via endpoint call to flask api, triggered from the
        #  experimentation python module (use secret key if spam protection is required, or implement time-out on
        #  end-point call in serving.)
        self._update_interval = model_update_interval
        self._update_thread = Thread(target=self._update_global_models, daemon=True)

        self._update_thread.start()

    def infer_from_global_model(self, data: dict, model_stage: str = 'production'):
        if not ('staging' in self._models and 'production' in self._models):
            return self._DEFAULT_RESPONSE

        # If specified model stage for the model to load is staging, keep it staging. If not, default to production.
        model = self._models['staging'] if model_stage.lower() == 'staging' else self._models['production']
        return self._infer_from_model(model, data)

    def infer_from_calibrated_model(self, model_id: str, data: dict):
        # If model id isn't loaded, create a thread and load model
        #   into memory. Use global inference until model is loaded.
        if model_id not in self._models:
            # Invoke model update on separate thread if not already invoked
            if model_id not in self._calibrated_model_retrieval_list:
                self._calibrated_model_retrieval_list.append(model_id)
                calibrated_model_retrieval = Thread(target=self._get_calibrated_model, args=[model_id], daemon=True)
                calibrated_model_retrieval.start()

            # Get inference from global model currently in production
            return self.infer_from_global_model(data=data)

        # Get loaded model from the model list
        model = self._models[f'{model_id}']
        return self._infer_from_model(model, data)

    @staticmethod
    def _infer_from_model(model: mlflow_pyfunc.PyFuncModel, data: dict):
        inputs = data.get('inputs')
        if type(inputs) is not list:
            raise TypeError('Cannot convert given input to multidimensional numpy array')

        try:
            data = np.asarray(inputs)
            result = model.predict(data)
        except InvalidArgument:
            raise TypeError('Input data did not follow expected format')

        return {k: v.tolist() for k, v in result.items()}

    def _update_global_models(self):
        messaging_manager.send("Checking for updated inference model", "inference")
        while self.do_keep_updating:
            try:
                staging_model_uuid = None
                if "staging" in self._models:
                    staging_model_uuid = self._models["staging"].metadata.model_uuid
                self._models['staging'] = mlflow_pyfunc.load_model(f'models:/{os.getenv("MODEL_NAME")}/Staging')
                if staging_model_uuid != self._models['staging'].metadata.model_uuid:
                    messaging_manager.send("Staging inference model updated", "inference")

                production_model_uuid = None
                if "production" in self._models:
                    production_model_uuid = self._models['production'].metadata.model_uuid
                self._models['production'] = mlflow_pyfunc.load_model(f'models:/{os.getenv("MODEL_NAME")}/Production')
                if production_model_uuid != self._models['production'].metadata.model_uuid:
                    messaging_manager.send("Production inference model updated", "inference")
            except MlflowException:  # If no model is available, do not update any further, wait and try again later.
                messaging_manager.send("No model found", "inference")
                pass

            time.sleep(self._update_interval)

    def _get_calibrated_model(self, model_id: str):
        # Search for calibrated model in mlflow runs
        calibrated_model = mlflow_search_runs(experiment_names=['ONNX Emotion Recognition'],
                                              filter_string=f'run_name = "{model_id}"')

        # If model has been found, load the model from its model run uri and add it to the model cache
        if len(calibrated_model) != 0:
            model_run_id = calibrated_model.iloc[0].loc['run_id']
            try:
                self._models[model_id] = mlflow_pyfunc.load_model(f'runs:/{model_run_id}/model')
                print('Calibrated model has been loaded.')
            except MlflowException:     # Occurs when calibrated model is still under development
                print('Model run exists, but the model does not exist.')
            except NoSuchFile:          # Occurs when the model has just been uploaded and is still processing
                print(f'No onnx model found in temporary directory.')

        # Mark model retrieval as being finished
        self._calibrated_model_retrieval_list.remove(model_id)
