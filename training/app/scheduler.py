from redis import Redis
from datetime import datetime
from collections import defaultdict
import json

import requests
from io import BytesIO
import zipfile
import psycopg2

from celery import Celery

redis_client = Redis(
    host="message-queue",
    port=6379,
    decode_responses=True
)

def register_tasks(celery_app: Celery):
    """
    Registers Celery tasks related to the training scheduler.

    This function defines and registers the `check_and_trigger_training` task with the provided Celery app.
    The task checks the length of a Redis queue containing labeled images. If the number of labeled items
    meets or exceeds the specified threshold, it groups items by dataset, uploads the curated dataset,
    triggers a training job, and cleans up processed items from Redis.

    Args:
        celery_app (Celery): The Celery application instance to register tasks with.

    The registered task accepts:
        threshold (int, optional): The minimum number of labeled items required to trigger training. Defaults to 50.
        **kwargs: Additional keyword arguments for task execution.

    Side Effects:
        - Reads from and writes to a Redis queue named "labeled_image_queue".
        - Uploads datasets and triggers training jobs when thresholds are met.
        - Removes processed items from the Redis queue.
    """
    @celery_app.task(name="app.scheduler.check_and_trigger_training")
    def check_and_trigger_training(threshold: int = 50, **kwargs):
        queue_name = "labeled_image_queue"
        queue_len = redis_client.llen(queue_name)  # Get the number of items in the Redis queue
        print(f"[SCHEDULER] Queue '{queue_name}' contains {queue_len} items.")

        if queue_len < threshold:
            # Not enough labeled items to trigger training
            print(f"[SCHEDULER] Not enough labels yet ({queue_len}/{threshold})")
            return

        # Retrieve all labeled items from the Redis queue
        labeled_items = [
            json.loads(redis_client.lindex(queue_name, i))
            for i in range(queue_len)
        ]

        # Group items by their dataset name
        datasets = defaultdict(list)
        for item in labeled_items:
            datasets[item["dataset_name"]].append(item)

        print("[SCHEDULER TEST] Dataset summary:")
        for dataset_name, items in datasets.items():
            print(f" - Dataset '{dataset_name}': {len(items)} labeled items")
            if len(items) >= threshold:
                # If enough items for this dataset, proceed with upload and training
                print(f"[SCHEDULER TEST] Threshold reached. Proceeding with training...")

                dataset_alias = upload_curated_dataset(dataset_name, items)
                if dataset_alias:
                    # Get user ID from the first datapoint in the batch
                    datapoint_id = items[0].get("datapoint_id")
                    user_id = get_user_from_db(datapoint_id) or "unknown"
                    # Trigger the training job
                    trigger_training(dataset_alias, user_id)

                    # Remove processed items from the Redis queue
                    for item in items:
                        redis_client.lrem(queue_name, 1, json.dumps(item))
                    print(f"[SCHEDULER] Cleaned up {len(items)} items from Redis.")
            else:
                # Not enough data for this dataset yet
                print(f"[SCHEDULER TEST] Not enough data yet for this dataset.")

def get_user_from_db(datapoint_id):
    """
    Retrieves the 'user' field from the metadata of a datapoint in the database.

    Args:
        datapoint_id: The ID of the datapoint to look up.

    Returns:
        The user associated with the datapoint, or None if not found or on error.
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host="postgres-db",
            dbname="chimp_database",
            user="chimp_user",
            password="chimp_password"
        )
        cursor = conn.cursor()
        # Query for the metadata of the datapoint with the given ID
        cursor.execute(
            "SELECT metadata FROM datapoints WHERE id = %s",
            (datapoint_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            metadata = result[0]
            # Return the 'user' field from the metadata dictionary
            return metadata.get("user")
    except Exception as e:
        # Log any errors that occur during the database operation
        print(f"[SCHEDULER][ERROR] Failed to retrieve user from DB: {e}")
    return None

def upload_curated_dataset(dataset_name, items):
    """
    Prepares and uploads a curated dataset as a ZIP archive.

    Args:
        dataset_name (str): The name of the dataset being curated.
        items (list): List of labeled items to include in the curated dataset.

    Returns:
        str or None: The alias of the uploaded dataset if successful, otherwise None.
    """
    print(f"[SCHEDULER] Preparing upload for curated dataset: {dataset_name}")
    zip_buffer = BytesIO()  # In-memory buffer for the ZIP file
    y = []                  # List to store labels
    metadata = []           # List to store metadata for each item

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host="postgres-db",
        dbname="chimp_database",
        user="chimp_user",
        password="chimp_password"
    )
    cursor = conn.cursor()

    # Create a ZIP file in memory and add each datapoint's image
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for item in items:
            try:
                datapoint_id = item["datapoint_id"]
                # Fetch the datapoint's image URL, label, and metadata from the database
                cursor.execute(
                    "SELECT x, y, metadata FROM datapoints WHERE id = %s",
                    (datapoint_id,)
                )
                result = cursor.fetchone()
                if not result:
                    continue
                x, y_label, meta = result
                filename = item["filename"]

                if x.startswith("https://datastore"):
                     x = x.replace("https://datastore", "http://datastore", 1)

                # Download the image from the URL and add it to the ZIP file
                response = requests.get(x)
                response.raise_for_status()
                zip_file.writestr(filename, response.content)

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

    zip_buffer.seek(0)  # Reset buffer pointer to the beginning
    # Generate a unique alias for the curated dataset
    dataset_alias = f"{dataset_name}_curated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    # Prepare files and data for upload
    files = {
        "file": (f"{dataset_alias}.zip", zip_buffer.getvalue()),
        "labels": (None, json.dumps(y)),
        "metadata": (None, json.dumps(metadata)),
        "dataset_name": (None, dataset_alias),
    }

    print(f"[SCHEDULER] Uploading curated dataset '{dataset_alias}' with {len(y)} items...")
    try:
        # Upload the curated dataset to the training API
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
            "basedata": "fer2013",
            "newdata": dataset_name,
            "personaldata": dataset_name,
            "experiment_name": experiment_name
        }

        r = requests.post("http://training-api:8000/tasks/run/Emotion+Recognition", data=form)
        print(f"[SCHEDULER] Trigger response: {r.status_code} - {r.text}")

    except Exception as e:
        print(f"[SCHEDULER][ERROR] Could not trigger training: {e}")
