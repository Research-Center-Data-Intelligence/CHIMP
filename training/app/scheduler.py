from redis import Redis
from datetime import datetime
from collections import defaultdict
import json
import zipfile
import psycopg2
from io import BytesIO
from PIL import Image
from celery import Celery
import requests
from minio.error import S3Error

from app.datastore import ManagedMinioDatastore

# Redis config
redis_client = Redis(
    host="message-queue",
    port=6379,
    decode_responses=True
)

# MinIO datastore correct initialiseren
datastore = ManagedMinioDatastore(
    access_key="minioadmin",
    secret_key="minioadmin",
    db_name="chimp_database",
    db_user="chimp_user",
    db_password="chimp_password"
)
datastore._datastore_uri = "datastore:9000"
datastore._database_uri = "postgres-db"
datastore._init_datastore()
datastore._init_database()

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
        conn = psycopg2.connect(
            host="postgres-db",
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

    conn = psycopg2.connect(
        host="postgres-db",
        dbname="chimp_database",
        user="chimp_user",
        password="chimp_password"
    )
    cursor = conn.cursor()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for item in items:
            try:
                datapoint_id = item["datapoint_id"]
                cursor.execute("SELECT x, y, metadata FROM datapoints WHERE id = %s", (datapoint_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"[SCHEDULER][WARNING] No datapoint found with id: {datapoint_id}")
                    continue

                x, y_label, meta = result
                filename = item["filename"]

                object_path = x.split("/datasets/")[-1]
                try:
                    response = datastore._client.get_object("datasets", object_path)
                    content = response.read()
                except S3Error as s3e:
                    print(f"[SCHEDULER][ERROR] Failed to fetch {object_path} from MinIO: {s3e}")
                    continue

                try:
                    # Open de afbeelding en converteer expliciet naar RGB
                    img = Image.open(BytesIO(content)).convert("RGB")
                    print(f"[DEBUG IMG] {filename} - size: {img.size}, mode: {img.mode}")

                    # Sla de geconverteerde afbeelding op naar een buffer
                    img_buffer = BytesIO()
                    img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)

                    # Voeg de RGB-afbeelding toe aan de zip
                    zip_file.writestr(filename, img_buffer.read())

                except Exception as img_debug_e:
                    print(f"[DEBUG IMG][ERROR] Cannot inspect or convert image {filename}: {img_debug_e}")

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
        r = requests.post("http://training-api:8000/managed_datasets", files=files)
        print(f"[SCHEDULER] Upload response: {r.status_code} - {r.text}")
        return dataset_alias if r.status_code == 200 else None
    except Exception as e:
        print(f"[SCHEDULER][ERROR] Upload failed: {e}")
        return None

def trigger_training(dataset_name: str, user_id: str):
    print(f"[SCHEDULER] Triggering training for curated dataset '{dataset_name}'")

    try:
        experiment_name = f"retrain_{dataset_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        form = {
            "user_id": user_id,
            "trainnew": "false",
            "basedata": "false",
            "newdata": "false",
            "personaldata": "true",
            "experiment_name": experiment_name,
            "base_model_name": "onnx_emo_datastore"
        }

        r = requests.post("http://training-api:8000/tasks/run/Emotion+Recognition", data=form)
        print(f"[SCHEDULER] Trigger response: {r.status_code} - {r.text}")

    except Exception as e:
        print(f"[SCHEDULER][ERROR] Could not trigger training: {e}")

