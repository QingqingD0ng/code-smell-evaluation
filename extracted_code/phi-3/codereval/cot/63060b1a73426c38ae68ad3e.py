import os
import yaml

def get_plugin_spec_flatten_dict(plugin_dir):
    spec_dict = {}
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as stream:
                    spec = yaml.safe_load(stream)
                    flatten_spec(spec, spec_dict, root)
    return spec_dict

def flatten_spec(spec, parent_key, root):
    for key, value in spec.items():
        full_key = f"{root}/{key}"
        if isinstance(value, dict):
            flatten_spec(value, full_key, root)
        else:
            spec_dict[full_key] = value