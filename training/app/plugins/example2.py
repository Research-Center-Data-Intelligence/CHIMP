from time import sleep
from typing import Any, Optional

from app.plugin import BasePlugin, PluginInfo


class Example2Plugin(BasePlugin):

    def __init__(self):
        self._info = PluginInfo(name="Example 2 Plugin", version="1.0")

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs) -> Optional[Any]:
        print(f"Start running {self._info.name}, sleeping for 20 seconds")
        sleep(20)
        print(f"End running {self._info.name}")
