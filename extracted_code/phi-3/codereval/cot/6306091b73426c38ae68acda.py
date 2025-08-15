class PluginsManager:
    def __init__(self, plugins_manager_service):
        self.plugins_manager_service = plugins_manager_service

    def get_plugins_manager(self):
        return self.plugins_manager_service


class ServiceRegistry:
    def _get_service(self, service_name):
        services = {
            'PLUGINS_MANAGER': PluginsManager(plugins_manager_service=PluginsManagerService())
        }
        return services.get(service_name)


class PluginsManagerService:
    pass