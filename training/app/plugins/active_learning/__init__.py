import os
import json
import numpy as np

# Keep GPU enabled by default; set FORCE_CPU=1 to explicitly disable it.
if os.getenv("FORCE_CPU", "0").strip().lower() in {"1", "true", "yes"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import glob
import datetime
from typing import Dict, Optional, List
from tensorflow.keras.models import load_model
from app.plugin import BasePlugin, PluginInfo
from .badge import BADGE

import mlflow
from mlflow.tracking import MlflowClient
from redis import Redis

##TODO: MV Make sure the Redis queue credentials work both in docker setup awa local debug
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

class ActiveLearningPlugin(BasePlugin):
    def __init__(self):
        self._info = PluginInfo(
            name="Active Learning",
            version="0.3",
            description="Selects samples from pool using BADGE and registers labeling task in PostgreSQL via datastore.",
            arguments={
                "experiment_name": {
                    "name": "experiment_name",
                    "type": "str",
                    "description": "Name of the MLFlow experiment to retrieve the model from.",
                    "optional": False,
                },
                "query_size": {
                    "name": "query_size",
                    "type": "int",
                    "description": "Number of samples to select from the pool.",
                    "optional": True
                },
                "pool_dataset": {
                    "name": "pool_dataset",
                    "type": "str",
                    "description": "Name of the unlabeled dataset to query from.",
                    "optional": False
                }
            },
            model_return_type=None
        )
        self.config: Dict = {}

    def init(self) -> PluginInfo:
        print("Initializing ActiveLearningPlugin")
        return self._info

    def run(self, *args, **kwargs) -> Optional[List[int]]:
        experiment_name = kwargs["experiment_name"]
        temp_dir = kwargs["temp_dir"]
        query_size = int(kwargs.get("query_size", 100))
        dataset_name = kwargs["pool_dataset"]

        print(f"Running plugin with arguments: {kwargs}")

        with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
            self.config = json.load(f)

        self._datastore.load_folder_to_filesystem(dataset_name, temp_dir, bucket="manageddataset")
        pool_dir = temp_dir

        model_dir = self._connector.get_artifact(
            save_to=os.path.join(temp_dir, "emotion_model"),
            model_name=experiment_name,
            experiment_name=experiment_name,
            artifact_path="keras"
        )

        keras_path = glob.glob(os.path.join(model_dir, "*.keras"))
        if not keras_path:
            raise RuntimeError("Geen .keras model gevonden in de map.")

        model = load_model(keras_path[0])

        X_pool, filenames = self.load_images_from_folder(pool_dir)
        if len(X_pool) == 0:
            raise RuntimeError("Er zijn geen afbeeldingen gevonden in de pool.")

        X_pool = X_pool / 255.0

        badge = BADGE(model=model, pool_dataset=X_pool, batch_size=32, num_samples=query_size)
        selected_indices = [int(i) for i in badge.select()]

        selection_data = {
            "selected_indices": selected_indices,
            "selected_filenames": [filenames[i] for i in selected_indices],
            "timestamp": datetime.datetime.now().isoformat(),
            "total": len(selected_indices)
        }

        selection_dir = os.path.join(temp_dir, "selection")
        os.makedirs(selection_dir, exist_ok=True)

        with open(os.path.join(selection_dir, "selection.json"), "w") as f:
            json.dump(selection_data, f)

        # TODO: MV check if still needed
        self._datastore.store_file_or_folder(
            target_path=os.path.join(dataset_name, "selection"),
            src_path=selection_dir
        )

        # TODO: MV check if still needed
        # Redis caching
        redis_key = f"labeling_task:{dataset_name}"
        redis_client.set(redis_key, json.dumps(selection_data))

        # PostgreSQL registratie (enkel noodzakelijke velden)
        self._datastore.store_labeling_task(
            dataset_id=dataset_name,
            total_images=len(selection_data["selected_filenames"]),
            num_labeled=0,
            status="pending",
            selection=selection_data["selected_filenames"]
        )

        print(f"[INFO] Labeling task opgeslagen in Redis en PostgreSQL voor dataset: {dataset_name}")
        return selected_indices

    def load_images_from_folder(self, folder_path):
        import cv2
        image_list = []
        filenames = []
        h, w = self.config["image_height"], self.config["image_width"]

        paths = glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True)
        paths += glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True)
        paths += glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)
        paths.sort()

        for path in paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARNING] Kon afbeelding niet laden: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (w, h))
                image_list.append(img)
                filenames.append(os.path.relpath(path, folder_path))
            except Exception as e:
                print(f"[ERROR] Fout bij verwerken afbeelding {path}: {e}")

        return np.array(image_list), filenames
