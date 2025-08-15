class PluginsManager:

    @classmethod

    def plugins_manager(cls):

        service_manager = cls._get_service()

        return service_manager.PLUGINS_MANAGER