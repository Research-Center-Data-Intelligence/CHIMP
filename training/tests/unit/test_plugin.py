from app.plugin import PluginLoader


class TestPluginLoader:
    """Tests for the plugin loader class."""

    def test_plugin_load_plugins(self, plugin_loader: PluginLoader, plugin: str):
        """Test the load_plugin method."""
        assert plugin not in [p["name"] for p in plugin_loader.loaded_plugins()]
        plugin_loader.load_plugins()
        assert plugin in [p["name"] for p in plugin_loader.loaded_plugins()]
