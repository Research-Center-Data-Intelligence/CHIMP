import importlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from flask import Flask
from typing import Any, Dict, List, Optional, Type

from app.connectors import BaseConnector


@dataclass
class PluginInfo:
    """PluginInfo contains the information about a plugin.

    Attributes
    ----------
    name : str
        The name of the plugin
    version : str
        The version of the plugin
    description : str
        Description of the plugin
    arguments : Dict[str, Dict[str, Union[str, Type]]]
        A dictionary with possible arguments for the plugin. Each argument has a key,
        which is the name of the (keyword) argument as passed to the plugin. The value
        is a dictionary containing three required fields and can contain one optional field:
            - name (a string with the name of the argument)
            - type (description of the type of the argument, as a string)
            - description (a description of the argument)
            - [OPTIONAL] optional (True if the argument is optional)
    model_return_type : Optional[str]
        The type of model object returned by the plugin. Preferably, this should be the value
        of one of the ModelType types. If the plugin does not return anything (including
        creating a model and uploading it via a connector), the model_return_type should
        be set to None.
    """

    name: str
    version: str
    description: str
    arguments: Dict[str, Dict[str, str]]
    model_return_type: Optional[str] = None


class BasePlugin(ABC):
    """This abstract class provides a base for implementing different plugins."""

    _info: PluginInfo
    _connector: BaseConnector

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
    def run(self, *args, **kwargs) -> Optional[Any]:
        """Run the plugin."""
        pass

    def info(self) -> dict:
        """Return the info for the plugin.

        Returns
        -------
        A dictionary containing the info about the plugin.
        """
        return asdict(self._info)


class PluginLoader:
    _app: Flask = None
    plugin_directory: str = ""
    _loaded_plugins: Dict[str, BasePlugin] = {}
    _connector: BaseConnector

    def init_app(self, app: Flask, connector: BaseConnector):
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize this extension with.
        connector : BaseConnector
            Connector instance provided to the plugins for storing models and metrics.

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
        self._connector = connector
        self.plugin_directory = app.config["PLUGIN_DIRECTORY"]

    def load_plugins(self):
        """Load all plugins from the configured plugin folder."""
        self._loaded_plugins = {}
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
                        plg._connector = self._connector
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

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name.

        Parameters
        ----------
        name : str
            The name of the plugin to load

        Returns
        -------
        The loaded plugin or None
        """
        return self._loaded_plugins.get(name)
