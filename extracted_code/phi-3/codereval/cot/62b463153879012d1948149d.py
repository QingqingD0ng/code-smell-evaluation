import os

def _group_files_by_xml_filename(files):
    # Implementation for grouping files by XML filename
    pass

def _explore_folder(folder):
    grouped_files = {}
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.lower().endswith('.xml'):
            grouped_files = _group_files_by_xml_filename([entry.path])
    return grouped_files

folder_path = '/path/to/your/folder'
grouped_files = _explore_folder(folder_path)