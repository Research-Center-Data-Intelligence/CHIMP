import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))

# First load generic .env file, then load a specific .env file for
# this component.
load_dotenv(os.path.join(basedir, "../.env"))
load_dotenv(os.path.join(basedir, "./.env"))

TESTING = os.environ.get("TESTING") or False
DEVELOPMENT = os.environ.get("DEVELOPMENT") or False
DEV = DEVELOPMENT

LEGACY_MODEL_NAME = os.environ.get("LEGACY_MODEL_NAME") or "onnx emotion model"
MODEL_UPDATE_INTERVAL_SECONDS = os.environ.get("MODEL_UPDATE_INTERVAL_SECONDS") or 300
