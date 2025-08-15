def _group_files_by_xml_filename(source, xmls, files):
    grouped_files = {xml: [] for xml in xmls}
    for file in files:
        file_extension = file.split('.')[-1].lower()
        if file_extension in grouped_files:
            grouped_files[file_extension].append(file)
    return grouped_files