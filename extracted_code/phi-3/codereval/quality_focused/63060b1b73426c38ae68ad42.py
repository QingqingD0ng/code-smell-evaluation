import os

class ConfigInjector:
    def __init__(self, ansible_config_path):
        self.ansible_config_path = ansible_config_path

    def inject_config(self):
        if 'ANSIBLE_CONFIG' not in os.environ:
            os.environ['ANSIBLE_CONFIG'] = self.ansible_config_path