from time import sleep
from typing import Any, Dict, Optional

from app.plugin import BasePlugin, PluginInfo


class Example2Plugin(BasePlugin):

    def __init__(self):
        self._info = PluginInfo(
            name="Example 2 Plugin",
            version="1.0",
            description="This is the most basic example plugin.",
            arguments={
                "start_value": {
                    "name": "start_value",
                    "type": "int",
                    "description": "The starting value",
                },
                "settings": {
                    "name": "settings",
                    "type": "Dict[str, str]",
                    "description": "The setting to use",
                    "optional": True,
                },
            },
            # datasets={
            #     "dataset": {"name": "dataset", "description": "Basic dataset"},
            #     "optional_ds": {
            #         "name": "optional_ds",
            #         "description": "An optional dataset",
            #         "optional": True,
            #     },
            # },
            model_return_type=None,
        )

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs) -> Optional[Any]:
        print(f"Start running {self._info.name}")
        print(f"Starting value: {kwargs['start_value']}")
        if "settings" in kwargs:
            print(f"Settings: {kwargs['settings']}")
        print("Sleeping for 20 seconds")
        sleep(20)
        print(f"End running {self._info.name}")
