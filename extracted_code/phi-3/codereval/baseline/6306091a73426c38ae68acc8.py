from configparser import ConfigParser

def list_of_file_names(settings_dirs, spec_option):
    parser = ConfigParser()
    file_list = []

    for directory in settings_dirs:
        config_file = f"{directory}/{spec_option}.ini"
        try:
            parser.read(config_file)
            file_list.append(config_file)
        except FileNotFoundError:
            pass

    return file_list