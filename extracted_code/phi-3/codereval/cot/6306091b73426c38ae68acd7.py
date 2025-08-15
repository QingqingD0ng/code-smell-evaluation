class ServiceManager:
    def _get_service(self):
        return {
            'ANSIBLE_CONFIG_MANAGER': 'ansible_config_manager_instance'
        }

class AnsibleConfigManager:
    def __init__(self, service_manager):
        self.service_manager = service_manager
        self.ansible_config_manager_instance = self._initialize_instance()

    def _initialize_instance(self):
        return 'ansible_config_manager'

    @classmethod
    def get_ansible_config_manager(cls, service_manager):
        service_details = service_manager._get_service()
        if 'ANSIBLE_CONFIG_MANAGER' in service_details:
            return cls(service_manager)._initialize_instance()
        else:
            raise ValueError("ANSIBLE_CONFIG_MANAGER not found in service details.")

# Usage
service_manager = ServiceManager()
ansible_config_manager = AnsibleConfigManager.get_ansible_config_manager(service_manager)