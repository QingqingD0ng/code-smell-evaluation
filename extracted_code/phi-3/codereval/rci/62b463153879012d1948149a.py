import os
from collections import defaultdict

def _group_files_by_xml_filename(source, xmls, files):
    grouped_files = defaultdict(list)
    for file in files:
        file_base_name, _ = os.path.splitext(file)
        for xml in xmls:
            if xml in file_base_name:
                grouped_files[xml].append(file)
                break
    return dict(grouped_files)