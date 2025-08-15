import zipfile

def _explore_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return [file_info.filename for file_info in zip_ref.infolist() if file_info.filename.lower().endswith('.xml')]

def _group_files_by_xml_filename(zip_path):
    return set(_explore_zipfile(zip_path))