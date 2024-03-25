"""The extensions module is responsible for creating objects for Flask extensions."""

from flask_cors import CORS

from .inference import InferenceManager

cors = CORS()
inference_manager = InferenceManager()
