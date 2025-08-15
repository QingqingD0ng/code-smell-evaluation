def select_filenames_by_prefix(prefix, files):
    matched_files = []
    for file in files:
        if file.startswith(prefix):
            matched_files.append(file)
    return matched_files