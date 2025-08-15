import zipfile

def _explore_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.xml'):
                yield file_info.filename

def _group_files_by_xml_filename(zip_path):
    xml_files = set()
    for xml_file in _explore_zipfile(zip_path):
        xml_files.add(xml_file)
    return xml_files