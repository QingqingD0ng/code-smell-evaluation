import zipfile

def _group_files_by_xml_filename(zip_path):
    groups = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.xml'):
                filename = file_info.filename.rsplit('/', 1)[-1]
                if filename in groups:
                    groups[filename].append(file_info.filename)
                else:
                    groups[filename] = [file_info.filename]
    
    return groups

def _explore_zipfile(zip_path):
    return _group_files_by_xml_filename(zip_path)