class WorkspaceManagerError(Exception):
    pass

class WorkspaceManager:
    WORKSPACE_MANAGER_SERVICE_NAME = 'WORKSPACE_MANAGER'

    def __init__(self):
        self._services = {}

    @classmethod
    def workspace_manager(cls):
        return cls()._lookup_service(cls.WORKSPACE_MANAGER_SERVICE_NAME)

    def _lookup_service(self, service_name):
        if service_name not in self._services:
            self._services[service_name] = self._mock_service_lookup(service_name)
        return self._services[service_name]

    def _mock_service_lookup(self, service_name):
        if service_name == self.WORKSPACE_MANAGER_SERVICE_NAME:
            return self._create_workspace_manager()
        raise WorkspaceManagerError(f"Service {service_name} not found")

    def _create_workspace_manager(self):
        # Mock WorkspaceManager creation, replace with actual implementation
        return WorkspaceManager()