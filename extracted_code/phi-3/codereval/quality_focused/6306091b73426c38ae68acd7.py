from typing import Type

class ServiceManager:
    _service_registry = {}

    @classmethod
    def register_service(cls, service_name, service_manager_cls):
        cls._service_registry[service_name] = service_manager_cls

    @classmethod
    def get_service(cls, service_name):
        service_manager_cls = cls._service_registry.get(service_name)
        if service_manager_cls:
            return service_manager_cls()
        raise ValueError(f"Service '{service_name}' not found")

class AnsibleConfigManager(ServiceManager):
    def get_config(self):
        # Implementation to retrieve the configuration
        pass

# Register the AnsibleConfigManager with the ServiceManager
ServiceManager.register_service('ANSIBLE_CONFIG_MANAGER', AnsibleConfigManager)

# Usage example
ansible_config_manager = ServiceManager.get_service('ANSIBLE_CONFIG_MANAGER')
config = ansible_config_manager.get_config()