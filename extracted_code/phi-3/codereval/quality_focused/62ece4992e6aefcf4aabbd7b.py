import os
import yaml

def write_configuration(config_filename, rendered_config, mode=0o600, overwrite=False):
    if os.path.exists(config_filename) and not overwrite:
        print(f"Error: Configuration file '{config_filename}' already exists. Set overwrite=True to overwrite it.")
        return

    with open(config_filename, 'w' if overwrite else 'x') as file:
        yaml.dump(rendered_config, file, default_flow_style=False)