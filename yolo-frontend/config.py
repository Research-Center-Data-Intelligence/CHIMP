import os


SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://localhost:5254")
TRAINING_API_URL = os.environ.get("TRAINING_API_URL", "http://localhost:5253")
MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolo_pose_demo")
MODEL_STAGE = os.environ.get("YOLO_MODEL_STAGE", "production")
MODEL_SESSION_ID = os.environ.get("YOLO_MODEL_SESSION_ID", "")
DATASET_NAME = os.environ.get("YOLO_DATASET_NAME", "yolo_pose_demo")
IMAGE_SIZE = 640
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_FRAME_COUNT = 5