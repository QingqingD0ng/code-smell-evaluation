import os

def write_configuration(config_filename, rendered_config, mode=0o600, overwrite=False):
    if os.path.exists(config_filename) and not overwrite:
        return
    with open(config_filename, 'w' if overwrite else 'x') as file:
        file.write(rendered_config)