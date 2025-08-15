import os
import yaml

def get_plugin_spec_flatten_dict(plugin_dir):
    plugin_info = {}
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        plugin_info.update(data)
                    except yaml.YAMLError as exc:
                        print(f"Error parsing {file_path}: {exc}")
    return plugin_info