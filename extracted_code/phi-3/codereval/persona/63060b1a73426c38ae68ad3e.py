import os
import yaml

def get_plugin_spec_flatten_dict(plugin_dir):
    spec_dict = {}
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.yaml'):
                with open(os.path.join(root, file), 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        for key, value in data.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in flatten_dict(value).items():
                                    spec_dict[f"{key}.{sub_key}"] = sub_value
                            else:
                                spec_dict[key] = value
                    except yaml.YAMLError as exc:
                        print(f"Error parsing YAML file {file}: {exc}")
    return spec_dict

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)