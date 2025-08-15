import yaml
import os

def get_plugin_spec_flatten_dict(plugin_dir):
    flattened_dict = {}
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.yaml'):
                with open(os.path.join(root, file), 'r') as stream:
                    plugin_data = yaml.safe_load(stream)
                    for key, value in plugin_data.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flattened_dict[(root, key, sub_key)] = sub_value
                        else:
                            flattened_dict[(root, key)] = value
    return flattened_dict