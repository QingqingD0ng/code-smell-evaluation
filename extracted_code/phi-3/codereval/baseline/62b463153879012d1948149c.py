def select_filenames_by_prefix(prefix, files):
    matching_files = []
    for file in files:
        if file.startswith(prefix):
            matching_files.append(file)
    return matching_files