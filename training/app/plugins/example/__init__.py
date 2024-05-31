from app.plugin import BasePlugin, PluginInfo

from . import additional_class


class ExamplePlugin(BasePlugin):

    def __init__(self):
        self._info = PluginInfo(
            name="Example Plugin",
            version="1.0",
            description="This example plugin uses a folder (module) instead of a single py file.",
            arguments={},
            datasets={},
            model_return_type="tensorflow",
        )

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs):
        print(f"Running {self._info.name}")
        return additional_class.AdditionalClass.some_method()
