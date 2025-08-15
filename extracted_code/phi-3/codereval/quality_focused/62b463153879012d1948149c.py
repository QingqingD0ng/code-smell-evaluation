def select_filenames_by_prefix(prefix, files):
    filtered_files = [file for file in files if file.startswith(prefix)]
    return filtered_files