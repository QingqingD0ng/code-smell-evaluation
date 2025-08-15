import yaml
import os

def write_configuration(config_filename, rendered_config, mode=0o600, overwrite=False):
    if os.path.exists(config_filename) and not overwrite:
        raise FileExistsError(f"File '{config_filename}' already exists and overwrite is set to False.")
    with open(config_filename, 'w' if not overwrite else 'x', mode) as file:
        yaml.dump(rendered_config, file)