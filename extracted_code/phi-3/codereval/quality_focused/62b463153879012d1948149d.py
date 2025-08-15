import os

def _explore_folder(folder):
    grouped_files = _group_files_by_xml_filename(folder)
    return grouped_files

def _group_files_by_xml_filename(folder):
    xml_files = {}
    for filename in os.listdir(folder):
        if filename.endswith('.xml'):
            filepath = os.path.join(folder, filename)
            xml_files.setdefault(filename, []).append(filepath)
    return xml_files