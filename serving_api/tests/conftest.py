import mlflow
import numpy as np
import os
import pytest
import shutil
from flask import Flask
from flask.testing import FlaskClient
from typing import List

from app.connectors import BaseConnector
from app.inference import InferenceManager
from app.model import BaseModel, OnnxModel

testdir = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def calibrated_model_id() -> str:
    return "test_calibrated_model_id"


@pytest.fixture
def calibrated_model_name() -> str:
    return "test_calibrated_model"


@pytest.fixture
def global_model_id() -> str:
    return "test_global_model_id"


@pytest.fixture
def global_model_name() -> str:
    return "test_global_model"


@pytest.fixture
def mocked_mlflow(
    monkeypatch,
    calibrated_model_id,
    calibrated_model_name,
    global_model_id,
    global_model_name,
):
    from app import connector as original_connector, connectors

    class MockModelImpl:
        inputs = [[[], ["float"]], []]

    class MockModelMetaData:
        def to_dict(self):
            return {"meta": "data"}

    class MockModel:
        _model_impl = MockModelImpl()
        metadata = MockModelMetaData()

        def predict(self, *args, **kwargs):
            return {"dense_3": np.array([0.53, 0.001, 0.04, 0.6, 0.02, 0.2, 0.12])}

    class MockMlflowClient:
        def get_run(self, run_id):
            class InfoObject:
                info: object
                run_name: str

            if run_id in [calibrated_model_id, global_model_id]:
                obj = InfoObject()
                info_object = InfoObject()
                info_object.run_name = calibrated_model_name
                obj.info = info_object
                return obj

        def search_model_versions(self, text):
            class ModelObject:
                name: str
                run_id: str

            cal_model_obj = ModelObject()
            cal_model_obj.name = calibrated_model_name
            cal_model_obj.run_id = calibrated_model_id
            glob_model_obj = ModelObject()
            glob_model_obj.name = global_model_name
            glob_model_obj.run_id = global_model_id
            return [cal_model_obj, glob_model_obj]

    def mock_search_runs(search_all_experiments=False, filter_string=""):
        class RunObj:
            run_id: str
            iloc: List = []

            def __len__(self):
                return len(self.iloc)

        if filter_string == f"run_name = {calibrated_model_name}":
            obj = RunObj()
            run_obj = RunObj()
            run_obj.run_id = calibrated_model_id
            obj.iloc = [run_obj]
            return obj
        return []

    def mock_load_model(uri):
        titles = [
            f"models:/{global_model_name}/staging",
            f"models:/{global_model_name}/production",
            f"runs:/{calibrated_model_id}/model",
        ]
        if uri in titles:
            return MockModel()
        raise mlflow.MlflowException("")

    def mock_init_connector():
        original_connector._client = MockMlflowClient()

    monkeypatch.setattr(connectors.mlflow, "search_runs", mock_search_runs)
    monkeypatch.setattr(connectors.mlflow_pyfunc, "load_model", mock_load_model)
    monkeypatch.setattr(original_connector, "_init_connector", mock_init_connector)


@pytest.fixture
def app(mocked_mlflow) -> Flask:
    from app import create_app, config

    config.TESTING = True
    app = create_app(config)

    ctx = app.app_context()
    ctx.push()

    yield app

    ctx.pop()


@pytest.fixture
def client(app) -> FlaskClient:
    return app.test_client()


@pytest.fixture
def connector(app) -> BaseConnector:
    return app.extensions.get("connector")


@pytest.fixture
def model(connector: BaseConnector, global_model_name: str) -> BaseModel:
    return connector.get_model(global_model_name)


@pytest.fixture
def inference_manager(app) -> InferenceManager:
    return app.extensions.get("inference_manager")


@pytest.fixture(scope="session", autouse=True)
def cleanup_mlruns_folder():
    yield

    # execute after last test
    if os.path.exists(os.path.join(testdir, "mlruns")):
        shutil.rmtree(os.path.join(testdir, "mlruns"))
