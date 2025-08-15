import yaml
import os
from collections import defaultdict

def flatten_dict(d, parent_key=(), sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_plugin_spec_flatten_dict(plugin_dir):
    plugin_info = defaultdict(dict)
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.yaml'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as stream:
                    plugin_data = yaml.safe_load(stream)
                    flattened_plugin_data = flatten_dict(plugin_data)
                    for key, value in flattened_plugin_data.items():
                        plugin_info[root][key] = value
    return plugin_info