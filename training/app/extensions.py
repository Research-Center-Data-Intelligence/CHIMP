from flask_cors import CORS

from app.plugin import PluginLoader
from app.worker import WorkerManager

cors = CORS()
plugin_loader = PluginLoader()
worker_manager = WorkerManager()
