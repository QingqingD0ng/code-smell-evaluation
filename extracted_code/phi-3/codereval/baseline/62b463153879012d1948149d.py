import os

def _explore_folder(folder):
    grouped_files = _group_files_by_xml_filename(folder)
    return grouped_files

def _group_files_by_xml_filename(directory):
    xml_files_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory, filename)
            xml_files_dict.setdefault(filename, []).append(file_path)
    return xml_files_dict