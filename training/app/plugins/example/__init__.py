from app.plugin import BasePlugin, PluginInfo

from . import additional_class


class ExamplePlugin(BasePlugin):

    def __init__(self):
        self._info = PluginInfo(name="Example Plugin", version="1.0")

    def init(self) -> PluginInfo:
        return self._info

    def run(self):
        print(f"Running {self._PluginInfo.name}")
        print(additional_class.AdditionalClass.some_method())
