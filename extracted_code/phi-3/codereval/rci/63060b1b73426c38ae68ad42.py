import os


class ConfigInjector:
    def __init__(self, ansible_config_path):
        self.ansible_config_path = ansible_config_path

    def inject_config(self):
        if not os.environ.get('ANSIBLE_CONFIG'):
            if isinstance(self.ansible_config_path, str) and self.ansible_config_path:
                os.environ['ANSIBLE_CONFIG'] = self.ansible_config_path
            else:
                raise ValueError("Invalid ansible_config_path provided.")


injector = ConfigInjector("/path/to/ansible/config")

injector.inject_config()