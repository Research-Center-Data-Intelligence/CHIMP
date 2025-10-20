from redis import Redis
from datetime import datetime
from collections import defaultdict
import json

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

import requests
import zipfile
import psycopg2
import os

from celery import Celery

from app.extensions import datastore


# Redis config
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", "6379"),
    decode_responses=True
)

# # MinIO datastore correct initialiseren
# datastore = ManagedMinioDatastore(
#     access_key="minioadmin",
#     secret_key="minioadmin",
#     db_name="chimp_database",
#     db_user="chimp_user",
#     db_password="chimp_password"
# )
# datastore._datastore_uri = "datastore:9000"
# datastore._database_uri = "postgres-db"
# datastore._init_datastore()
# datastore._init_database()

def register_tasks(celery_app: Celery):
    @celery_app.task(name="app.scheduler.check_and_trigger_training")
    def check_and_trigger_training(threshold: int = 50, **kwargs):
        queue_name = "labeled_image_queue"
        queue_len = redis_client.llen(queue_name)
        print(f"[SCHEDULER] Queue '{queue_name}' contains {queue_len} items.")

        if queue_len < threshold:
            print(f"[SCHEDULER] Not enough labels yet ({queue_len}/{threshold})")
            return

        labeled_items = [
            json.loads(redis_client.lindex(queue_name, i))
            for i in range(queue_len)
        ]

        datasets = defaultdict(list)
        for item in labeled_items:
            datasets[item["dataset_name"]].append(item)

        print("[SCHEDULER] Dataset summary:")
        for dataset_name, items in datasets.items():
            print(f" - Dataset '{dataset_name}': {len(items)} labeled items")
            if len(items) >= threshold:
                print(f"[SCHEDULER] Threshold reached. Proceeding with training...")

                dataset_alias = upload_curated_dataset(dataset_name, items)
                if dataset_alias:
                    datapoint_id = items[0].get("datapoint_id")
                    user_id = get_user_from_db(datapoint_id) or "unknown"
                    trigger_training(dataset_alias, user_id)

                    for item in items:
                        redis_client.lrem(queue_name, 1, json.dumps(item))
                    print(f"[SCHEDULER] Cleaned up {len(items)} items from Redis.")
            else:
                print(f"[SCHEDULER] Not enough data yet for this dataset.")

def get_user_from_db(datapoint_id):
    try:
        ## TODO MV: dont hardcode login here
        conn = psycopg2.connect(
            host=os.getenv("DATABASE_URI", "localhost"),
            dbname="chimp_database",
            user="chimp_user",
            password="chimp_password"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM datapoints WHERE id = %s", (datapoint_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            metadata = result[0]
            return metadata.get("user")
    except Exception as e:
        print(f"[SCHEDULER][ERROR] Failed to retrieve user from DB: {e}")
    return None

def upload_curated_dataset(dataset_name, items):
    print(f"[SCHEDULER] Preparing upload for curated dataset: {dataset_name}")
    zip_buffer = BytesIO()
    y = []
    metadata = []

    ## TODO MV: dont hardcode login here
    conn = psycopg2.connect(
        host=os.getenv("DATABASE_URI", "localhost"),
        dbname="chimp_database",
        user="chimp_user",
        password="chimp_password"
    )
    cursor = conn.cursor()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for item in items:
            try:
                ##TODO MV: do we want direct access to postgress here, or hide it behind the connectors interface?
                datapoint_id = item["datapoint_id"]
                cursor.execute("SELECT x, y, metadata FROM datapoints WHERE id = %s", (datapoint_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"[SCHEDULER][WARNING] No datapoint found with id: {datapoint_id}")
                    continue

                x, y_label, meta = result
                filename = item["filename"]

                # Extract bucket name and object path
                parsed_url = urlparse(x)
                path_parts = parsed_url.path.lstrip('/').split('/', 1)  # Remove leading slash and split into bucket and object path
                bucket_name = path_parts[0]  # First part is the bucket name
                object_path = path_parts[1]

                # Download the image from the URL and add it to the ZIP file
                response = datastore._client.get_object(bucket_name,object_path)
         
                zip_file.writestr(filename, response.read())

                # Collect the label and metadata for this datapoint
                y.append(y_label)
                metadata.append({
                    "user": meta.get("user", "unknown"),
                    "original_dataset": dataset_name
                })

            except Exception as e:
                print(f"[SCHEDULER][ERROR] Failed to add datapoint {item.get('datapoint_id')}: {e}")

    cursor.close()
    conn.close()

    zip_buffer.seek(0)
    dataset_alias = f"{dataset_name}_curated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    files = {
        "file": (f"{dataset_alias}.zip", zip_buffer.getvalue()),
        "labels": (None, json.dumps(y)),
        "metadata": (None, json.dumps(metadata)),
        "dataset_name": (None, dataset_alias),
    }

    print(f"[SCHEDULER] Uploading curated dataset '{dataset_alias}' with {len(y)} items...")
    try:
        # Upload the curated dataset to the training API
        r = requests.post(os.getenv("TRAINING_SERVER_URL") + "/managed_datasets", files=files)
        print(f"[SCHEDULER] Upload response: {r.status_code} - {r.text}")
        return dataset_alias if r.status_code == 200 else None
    except Exception as e:
        print(f"[SCHEDULER][ERROR] Upload failed: {e}")
        return None

def trigger_training(dataset_name: str, user_id: str):
    print(f"[SCHEDULER] Triggering training for curated dataset '{dataset_name}'")

    ## TODO MV: sent dataset_alias with request iso user_id, this call now traines on all data from user_id as defined in the plugin. the plugin should have functionality to train on a specific dataset?
    try:
        experiment_name = f"retrain_{dataset_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        form = {
            "user_id": user_id,
            "trainnew": "False",
            "basedata": "False",
            "newdata": "False",
            "personaldata": "True",
            "experiment_name": "onnx_emo_datastore"
        }

        r = requests.post(os.getenv("TRAINING_SERVER_URL") + "/tasks/run/Emotion+Recognition", data=form)
        print(r)
        print(f"[SCHEDULER] Trigger response: {r.status_code} - {r.text}")

    except Exception as e:
        print(f"[SCHEDULER][ERROR] Could not trigger training: {e}")

