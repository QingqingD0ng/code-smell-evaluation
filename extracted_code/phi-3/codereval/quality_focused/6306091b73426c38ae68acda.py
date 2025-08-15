class PluginsManager:
    @classmethod
    def get_plugin_manager(cls):
        service = cls._get_service()
        return service[cls.PLUGINS_MANAGER]

    @classmethod
    def PLUGINS_MANAGER(cls):
        return 'plugin_manager'

    @classmethod
    def _get_service(cls):
        # This method should be implemented to return a dictionary
        # with service names as keys and service instances as values.
        # For example:
        # return {'plugin_manager': PluginManager(), 'other_service': OtherService()}
        pass