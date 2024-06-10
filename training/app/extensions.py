from flask_cors import CORS

from app import config
from app.config import DATASTORE_ACCESS_KEY, DATASTORE_SECRET_KEY
from app.connectors import BaseConnector, MLFlowConnector
from app.datastore import BaseDatastore, MinioDatastore
from app.plugin import PluginLoader
from app.worker import WorkerManager

cors = CORS()
plugin_loader = PluginLoader()
worker_manager = WorkerManager()
connector: BaseConnector = MLFlowConnector()
datastore: BaseDatastore = MinioDatastore(DATASTORE_ACCESS_KEY, DATASTORE_SECRET_KEY)
