from flask_cors import CORS

from app.plugin import PluginLoader

cors = CORS()
plugin_loader = PluginLoader()
