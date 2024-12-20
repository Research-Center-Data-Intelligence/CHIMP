import json
import os
import cv2
import numpy as np
import tensorflow as tf
import tf2onnx
from typing import Dict, Optional
from tensorflow.python.keras.models import save_model as tf_save_model
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

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
                "trainnew": {
                    "name": "trainnew",
                    "type": "bool",
                    "description": "Whether to train a new model, or to finetune an existing one",
                    "optional": True,
                },
                "basedata": {
                    "name": "basedata",
                    "type": "bool",
                    "description": "whether to use a base (exeternal/foudational) database as part of the training",
                    "optional": True,
                },
                "newdata": {
                    "name": "newdata",
                    "type": "bool",
                    "description": "whether to use all new (user) aquired data as part of the training",
                    "optional": True,
                },
                "personaldata": {
                    "name": "personaldata",
                    "type": "bool",
                    "description": "whether to use only specific person data as part of the training",
                    "optional": True,
                },
                "user_id": {
                    "name": "user_id",
                    "type": "str",
                    "description": "user_id in case personal data is used",
                    "optional": True,
                }
            },
            model_return_type="onnx",
        )

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs) -> Optional[str]:
        with open(os.path.join(plugin_dir, "config.json")) as f:
            self.config = json.load(f)
        
        #convert to actual booleans
        for key, value in kwargs.items():
            if value == 'True':
                kwargs[key] = True
            elif value == 'False':
                kwargs[key] = False
        #MV TODO: implement training and model storage, including inserting the dataset postgres table entries
        # return "training still needs to be implemented"

        #MV TODO: built logic based on plugin info
        # 1) New Model Personal Data
        if kwargs["trainnew"] & kwargs["personaldata"] & (not kwargs["basedata"]) & (not kwargs["newdata"]):
                pass

        # 2) New Model Base data + Personal Data
        if kwargs["trainnew"] & kwargs["personaldata"] & kwargs["basedata"] & (not kwargs["newdata"]):
            pass

        # 3) New Model Base data + All user Data
        if kwargs["trainnew"] & (not kwargs["personaldata"]) & kwargs["basedata"] & kwargs["newdata"]:
            pass

        # 4) Fine tune on Personal Data
        if (not kwargs["trainnew"]) & kwargs["personaldata"] & (not kwargs["basedata"]) & (not kwargs["newdata"]):
            #TODO MV: add error checking to the retrieval
            #MV TODO: for finetuning do not get the data from datapoints but from dataset and the current model id
            # Select query
            kwargs["user_id"] = 'MV' #MV TODO: get correct username from frontend
            select_query = f"""SELECT * FROM datapoints
            WHERE metadata->>'user' = '{kwargs["user_id"]}'
            LIMIT 30;
            """

            rows = self.load_data(select_query)
            model_path = self._connector.get_artifact(os.path.join(kwargs["temp_dir"],"basemodel"), model_name=kwargs["experiment_name"], experiment_name=kwargs["experiment_name"], artifact_path="keras")
            print(model_path)
            emotion_calibrationmodel_generator = EmotionModelCalibrator(self.config, model_path, self.data)
            tf_model, history = emotion_calibrationmodel_generator.generate()[0]
            run_name=kwargs["run_name"] = "calib_" + kwargs["run_name"]
        
        # 5) Fine tune on all user Data
        if (not kwargs["trainnew"]) & (not kwargs["personaldata"]) & (not kwargs["basedata"]) & kwargs["newdata"]:
            pass


        input_sig = [tf.TensorSpec(tf_model.input_spec[0].shape,tf.float32)]
        tf_model.output_names = ["output"]
        # Upload  tensorflow version using temporary directory as artifact
        # Note: Important for calibration purposes. (assets folder does not get uploaded, as it is empty)
        tf_path = os.path.join(kwargs["temp_dir"], "tensorflow")
        os.mkdir(tf_path)
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
        run_name, run_id = self._connector.store_model(
            experiment_name=kwargs["experiment_name"],
            model_name=kwargs["experiment_name"],
            run_name=run_name,
            model=onnx_model,
            model_type="onnx",
            hyperparameters=hyperparameters,
            metrics=metrics,
            artifacts= {'tensorflow' : tf_path}, #save the tensorflow version as well
        )

        with self._datastore._db_conn.cursor() as cursor:
            insert_data = []
            #MV TODO: log the used data and run_id to the dataset table.
            for row in rows:
                datapoint_id = row[0]
                insert_data.append((datapoint_id, run_id)) 

            insert_query = """INSERT INTO dataset (datapoint_id, run_id) VALUES (%s, %s);"""
            
            cursor.executemany(insert_query, insert_data)
            self._datastore._db_conn.commit()

        return run_name

    def load_data(self, select_query):
        """Loads the emotion image data from the data folder into memory. 
        Do Not preprocess the data, leave this up to model.py"""
        idx_label = {class_ : category for class_, category in enumerate(self.config["categories"])}
        label_idx = {category : class_ for class_, category in enumerate(self.config["categories"])}
        self.data = {"image_data": [], "class_": [], "category": []}

        with self._datastore._db_conn.cursor() as cursor:
            cursor.execute(select_query)

            # Fetch all rows
            rows = cursor.fetchall()

            # Print each row
            for row in rows:
                print(f"ID: {row[0]}, X: {row[1]}, Y: {row[2]}, Metadata: {row[3]}")
                opath = row[1]
                parsed_url = urlparse(opath)

                # Extract bucket name and object path
                path_parts = parsed_url.path.lstrip('/').split('/', 1)  # Remove leading slash and split into bucket and object path
                bucket_name = path_parts[0]  # First part is the bucket name
                object_path = path_parts[1]

                response = self._datastore._client.get_object(bucket_name,object_path)
                image = Image.open(BytesIO(response.read()))

                response.close()
                response.release_conn()
                self.data["image_data"].append(image)
                self.data["class_"].append(label_idx[row[2]])
                self.data["category"].append(row[2])

        self.data["image_data"] = np.array(self.data["image_data"])
        self.data["class_"] = np.array(self.data["class_"])
        self.data["category"] = np.array(self.data["category"])

        #return the table with the query results
        return rows