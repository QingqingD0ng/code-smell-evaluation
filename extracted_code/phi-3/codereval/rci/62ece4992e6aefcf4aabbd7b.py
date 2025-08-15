import os

import yaml


def file_exists_and_not_overwrite(filename, overwrite):
    return os.path.exists(filename) and not overwrite


def write_yaml(file_name, content, file_mode):
    with open(file_name, 'w' if file_mode is None else file_mode) as file:
        yaml.dump(content, file)


def write_configuration(config_file, config_data, file_mode=0o600, overwrite=False):
    if not file_exists_and_not_overwrite(config_file, overwrite):
        write_yaml(config_file, config_data, file_mode)