class ServiceManager:
    # Assuming the existence of cls._get_service() method
    @classmethod
    def _get_service(cls):
        # Placeholder for the actual implementation that retrieves the service
        pass

    @classmethod
    def ansible_config_manager(cls):
        service = cls._get_service()
        return service.ANSIBLE_CONFIG_MANAGER

# Example usage:
# Assuming there is a service instance with the attribute 'ANSIBLE_CONFIG_MANAGER'
# service_instance = ServiceManager()
# config_manager = ServiceManager.ansible_config_manager()