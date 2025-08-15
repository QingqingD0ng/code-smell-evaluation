from typing import List

def list_of_file_names(settings_dirs: List[str], spec_option: 'SpecOption') -> List[str]:
    file_names = []
    for directory in settings_dirs:
        # Assuming SpecOption is a callable that can extract file names from directories
        file_names.extend(spec_option(directory))
    return file_names