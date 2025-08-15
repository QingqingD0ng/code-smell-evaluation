class PluginsManager:
    pass

class SomeService:
    PLUGINS_MANAGER = PluginsManager()

    def _get_service(self):
        # This method should return the service object
        # For example purposes, we'll return a mock service with PLUGINS_MANAGER attribute
        return self

def plugins_manager(cls):
    service = cls._get_service()
    return vars(service).get('PLUGINS_MANAGER', None)