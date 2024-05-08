import importlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from flask import Flask
from typing import Dict, List


@dataclass
class PluginInfo:
    name: str
    version: str


class BasePlugin(ABC):
    """This abstract class provides a base for implementing different plugins."""

    _info: PluginInfo

    @abstractmethod
    def init(self) -> PluginInfo:
        """Run any initialization code for initializing the plugin, this includes at least
        providing a PluginInfo object and storing it in self._info.

        Returns
        -------
        The created PluginInfo object.
        """
        pass

    @abstractmethod
    def run(self):
        """

        Returns
        -------

        """
        pass

    def info(self) -> dict:
        return asdict(self._info)


class PluginLoader:
    _app: Flask = None
    plugin_directory: str = ""
    _loaded_plugins: Dict[str, BasePlugin] = {}

    def init_app(self, app: Flask):
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize this extension with.

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension has already been initialized
        """
        if "plugin_loader" in app.extensions:
            raise RuntimeError(
                "A 'plugin_loader' instance has already been registered on this Flask app."
            )
        app.extensions["plugin_loader"] = self
        self._app = app
        self.plugin_directory = app.config["PLUGIN_DIRECTORY"]

    def load_plugins(self):
        for file in os.listdir(self.plugin_directory):
            path = os.path.join(self.plugin_directory, file)
            module = None
            if os.path.isdir(path):
                if "__init__.py" in os.listdir(path):
                    module = importlib.import_module(f"app.plugins.{file}")
            elif file.endswith(".py"):
                module = importlib.import_module(f"app.plugins.{file[:-3]}")
            if module:
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BasePlugin)
                        and obj != BasePlugin
                    ):
                        plg = obj()
                        info = plg.info()
                        self._loaded_plugins[info["name"]] = plg

    def loaded_plugins(self, include_details: bool = False) -> List:
        """Get an overview of the loaded plugins.

        Parameters
        ----------
        include_details : Optional[bool]
            Whether to include all the details of each plugin or only the name, defaults to False

        Returns
        -------
        A list of loaded plugins.
        """
        return [
            (
                self._loaded_plugins[p].info()
                if include_details
                else self._loaded_plugins[p].info()["name"]
            )
            for p in self._loaded_plugins
        ]
