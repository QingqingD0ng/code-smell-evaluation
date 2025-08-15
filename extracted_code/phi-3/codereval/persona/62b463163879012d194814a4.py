import zipfile
import os

def _group_files_by_xml_filename(files_in_zip):
    xml_files = {}
    for file in files_in_zip:
        if file.endswith('.xml'):
            base_name = os.path.basename(file)
            xml_files[base_name] = file
    return xml_files

def _explore_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        xml_files = _group_files_by_xml_filename(zip_ref.namelist())
    return xml_files