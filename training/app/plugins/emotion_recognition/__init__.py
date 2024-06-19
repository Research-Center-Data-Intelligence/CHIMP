import json
import os
import cv2
import numpy as np
import tensorflow as tf
import tf2onnx
from typing import Dict, Optional
from tensorflow.python.keras.models import save_model as tf_save_model

from app.plugin import BasePlugin, PluginInfo

from .model import EmotionModelGenerator, EmotionModelCalibrator

plugin_dir = os.path.abspath(os.path.dirname(__file__))


class EmotionRecognitionPlugin(BasePlugin):
    config: Dict
    data: Dict
    data_dir: str
    calibration_dir: str

    def __init__(self):
        self._info = PluginInfo(
            name="Emotion Recognition",
            version="1.0",
            description="An emotion recognition model.",
            arguments={
                "experiment_name": {
                    "name": "experiment_name",
                    "type": "str",
                    "description": "Name of the MLFLOW Experiment to use",
                    "optional": False,
                },
                "calibrate": {
                    "name": "calibrate",
                    "type": "bool",
                    "description": "Whether to calibrate an existing model, if not set, a global model is trained",
                    "optional": True,
                },
                "calibration_id": {
                    "name": "calibration_id",
                    "type": "str",
                    "description": "ID to use to denote the calibrated model by",
                    "optional": True,
                },
            },
            datasets={
                "train": {"name": "train", "description": "Training dataset"},
                "calibration": {
                    "name": "calibration",
                    "description": "Calibration dataset",
                    "optional": True,
                },
            },
            model_return_type="onnx",
        )

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs) -> Optional[str]:
        with open(os.path.join(plugin_dir, "config.json")) as f:
            self.config = json.load(f)
            
        if "calibrate" in kwargs and kwargs["calibrate"]:
            if (
                "calibration" not in kwargs["datasets"]
                or "calibration_id" not in kwargs
            ):
                raise RuntimeError(
                    "If 'calibrate' is set to true, the 'calibration' dataset and 'calibration_id' field are required"
                )
            self.calibration_dir = os.path.join(kwargs["temp_dir"], "calibration")
            dataset_name = kwargs["datasets"]["calibration"]
            self._datastore.load_folder_to_filesystem(dataset_name, self.calibration_dir)

            print(f"calibration dataset: {self.calibration_dir}")
            print(f"calibration id: {kwargs['calibration_id']}")

            self.load_data(self.calibration_dir)
            #print("CALIBRATION NOT IMPLEMENTED! Creating new model instead")
            model_path = self._connector.get_artifact(os.path.join(kwargs["temp_dir"],"basemodel"), model_name=kwargs["experiment_name"], experiment_name=kwargs["experiment_name"], artifact_path="tensorflow")
            print(model_path)
            emotion_calibrationmodel_generator = EmotionModelCalibrator(self.config, model_path, self.data)
            tf_model, history = emotion_calibrationmodel_generator.generate()[0]
            run_name=kwargs["run_name"] = "calib_" + kwargs["run_name"]
        else:
            self.data_dir = os.path.join(kwargs["temp_dir"], "train")
            dataset_name = kwargs["datasets"]["train"]
            self._datastore.load_folder_to_filesystem(dataset_name, self.data_dir)

            self.load_data(self.data_dir)

            emotion_model_generator = EmotionModelGenerator(self.config, self.data)
            tf_model, history = emotion_model_generator.generate()[0]
            run_name=kwargs["run_name"] = "base_" + kwargs["run_name"]

        input_sig = [
            tf.TensorSpec(
                [None, self.config["image_height"], self.config["image_width"], 1],
                tf.float32,
            )
        ]
        tf_model.output_names = ["output"]
        # Upload  tensorflow version using temporary directory as artifact
        # Note: Important for calibration purposes. (assets folder does not get uploaded, as it is empty)
        os.mkdir(os.path.join(kwargs["temp_dir"], "tensorflow"))
        tf_path = os.path.join(kwargs["temp_dir"], "tensorflow")
        #tf_save_model(tf_model, tf_path) #does not work for keras 3
        tf_model.save(os.path.join(tf_path, "model.keras"))
        onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_sig, opset=13)
        metrics = {
            k: v[0]
            for k, v in history.history.items()
            if k in ("accuracy", "loss", "val_accuracy", "val_loss")
        }
        hyperparameters = {
            k: v
            for k, v in self.config.items()
            if k
            in (
                "learning_rate",
                "optimizer",
                "epochs",
                "batch_size",
                "convolutional_layers",
                "dense_layers",
                "early_stopping",
            )
        }
        run_name = self._connector.store_model(
            experiment_name=kwargs["experiment_name"],
            model_name=kwargs["experiment_name"],
            run_name=run_name,
            model=onnx_model,
            model_type="onnx",
            hyperparameters=hyperparameters,
            metrics=metrics,
            artifacts= {'tensorflow' : tf_path}, #save the tensorflow version as well
            tags={"dataset": dataset_name}
        )

        return run_name

    def load_data(self, data_dir):
        """Loads the emotion image data from the data folder into memory. Then process
        the data by reshaping the 'image_data' into a numpy array of (-1, <image height>,
        <image width>, <colours>), where image height and width are stored in the
        configuration file, and the amount of colors is assumed to be 1 (gray-scale).
        The pixels of each image are normalized to be a value between 0 and 1"""
        self.data = {"image_data": [], "class_": [], "category": []}

        for class_, category in enumerate(self.config["categories"]):
            directory = os.path.join(data_dir, "train", category)

            for image in os.listdir(directory):
                image_data = cv2.imread(
                    os.path.join(directory, image), cv2.IMREAD_GRAYSCALE
                )
                image_data = cv2.resize(
                    image_data,
                    (self.config["image_height"], self.config["image_width"]),
                )
                self.data["image_data"].append(image_data)
                self.data["class_"].append(class_)
                self.data["category"].append(category)

        reshaped_data = np.array(self.data["image_data"]).reshape(
            (-1, self.config["image_height"], self.config["image_width"], 1)
        )
        self.data["image_data"] = reshaped_data / 255
        self.data["class_"] = np.array(self.data["class_"])
        self.data["category"] = np.array(self.data["category"])
