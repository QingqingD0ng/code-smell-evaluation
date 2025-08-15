class PluginsManager:
    def __init__(self):
        self._plugins = {}

    def register_plugin(self, name, plugin):
        self._plugins[name] = plugin

    def get_plugin(self, name):
        return self._plugins.get(name)


class SomeService:
    _PLUGINS_MANAGER = PluginsManager()

    def _get_service(self):
        return self


class Application:
    def __init__(self):
        self.service = SomeService()

    def plugins_manager(self):
        return self.service._PLUGINS_MANAGER


# Example usage
app = Application()
plugin_manager = app.plugins_manager()

plugin_manager.register_plugin('example_plugin', SomePlugin())
example_plugin = plugin_manager.get_plugin('example_plugin')