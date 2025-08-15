def select_filenames_by_prefix(prefix, files):
    matching_files = [file for file in files if file.startswith(prefix)]
    return matching_files