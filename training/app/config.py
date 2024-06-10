import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))

# First load generic .env file, then load a specific .env file
# for this component.
load_dotenv(os.path.join(basedir, "../../.env"))
load_dotenv(os.path.join(basedir, "../.env"))

TESTING = os.environ.get("TESTING")
DEVELOPMENT = os.environ.get("DEVELOPMENT") or False
DEV = DEVELOPMENT

LEGACY_PLUGIN_NAME = os.environ.get("LEGACY_PLUGIN_NAME") or "Emotion Recognition"
LEGACY_DATASET_NAME = os.environ.get("LEGACY_DATASET_NAME") or "emotions"

TRACKING_URI = os.environ.get("TRACKING_URI") or "http://localhost:8999"

PLUGIN_DIRECTORY = os.environ.get("PLUGIN_DIRECTORY") or os.path.join(
    basedir, "./plugins"
)
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY") or os.path.join(
    basedir, "../datasets"
)

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL") or "amqp://localhost"
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND") or "rpc://localhost"

DATASTORE_URI = os.environ.get("DATASTORE_URI") or "localhost:9000"
DATASTORE_ACCESS_KEY = os.environ.get("DATASTORE_ACCESS_KEY") or ""
DATASTORE_SECRET_KEY = os.environ.get("DATASTORE_SECRET_KEY") or ""
