from flask_cors import CORS

from app.connectors import MLFlowConnector
from app.inference import InferenceManager

cors = CORS()
connector = MLFlowConnector()
inference_manager = InferenceManager()
