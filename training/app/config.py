import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))

# First load generic .env file, then load a specific .env file
# for this component.
load_dotenv(os.path.join(basedir, "../.env"))
load_dotenv(os.path.join(basedir, "./.env"))

TESTING = os.environ.get("TESTING")
DEVELOPMENT = os.environ.get("DEVELOPMENT") or False
DEV = DEVELOPMENT

LEGACY_MODEL_NAME = os.environ.get("LEGACY_MODEL_NAME") or "onnx emotion model"
TRACKING_URI = os.environ.get("TRACKING_URI") or "http://localhost:8999"

PLUGIN_DIRECTORY = os.environ.get("PLUGIN_DIRECTORY") or os.path.join(
    basedir, "./plugins"
)
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY") or os.path.join(
    basedir, "../datasets"
)

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL") or "amqp://localhost"
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND") or "rpc://localhost"
