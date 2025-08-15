from configparser import ConfigParser

def list_of_file_names(settings_dirs, spec_option='default'):
    config = ConfigParser()
    file_names = []

    for settings_dir in settings_dirs:
        config.read(settings_dir / f'{spec_option}.ini')
        for section in config.sections():
            for key, value in config.items(section):
                file_name = f"{settings_dir / section.replace('.', '_')}_{key}.ini"
                file_names.append(file_name)

    return file_names