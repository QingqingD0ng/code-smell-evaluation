from abc import ABCMeta, abstractmethod

class WorkspaceManager(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def _get_service(cls):
        pass

class WorkspaceManagerFactory:
    @classmethod
    def create_workspace_manager(cls, service_name):
        if service_name == "WORKSPACE_MANAGER":
            return WorkspaceManager()
        else:
            raise ValueError(f"No workspace manager found for service name: {service_name}")

# Usage
# Assuming WorkspaceManager is a concrete implementation of the abstract WorkspaceManager class
workspace_manager = WorkspaceManagerFactory.create_workspace_manager("WORKSPACE_MANAGER")