import zipfile

def _group_files_by_xml_filename(files):
    # Assuming this function groups files by their XML filename
    xml_groups = {}
    for file in files:
        if file.endswith('.xml'):
            xml_filename = file.split('/')[-1]
            if xml_filename not in xml_groups:
                xml_groups[xml_filename] = []
            xml_groups[xml_filename].append(file)
    return xml_groups

def _explore_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        xml_files = [file for file in zip_ref.namelist() if file.endswith('.xml')]
        grouped_xml_files = _group_files_by_xml_filename(xml_files)
        return grouped_xml_files