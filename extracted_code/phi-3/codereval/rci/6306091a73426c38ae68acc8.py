import os

def list_of_file_names(settings_dirs, spec_option):
    ini_file_paths = []

    for directory in settings_dirs:
        config_file_path = os.path.join(directory, f"{spec_option}.ini")
        if os.path.isfile(config_file_path):
            ini_file_paths.append(config_file_path)

    return ini_file_paths