from app.plugin import BasePlugin, PluginInfo


class Example2Plugin(BasePlugin):

    def __init__(self):
        self._info = PluginInfo(name="Example 2 Plugin", version="1.0")

    def init(self) -> PluginInfo:
        return self._PluginInfo

    def run(self):
        print(f"Running {self._PluginInfo.name}")
