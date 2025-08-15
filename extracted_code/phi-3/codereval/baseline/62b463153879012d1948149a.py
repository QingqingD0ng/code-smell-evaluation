import os
from collections import defaultdict

def _group_files_by_xml_filename(source, xmls, files):
    grouped_files = defaultdict(list)
    for file in files:
        base_name = os.path.splitext(file)[0]
        for xml in xmls:
            if xml in file:
                grouped_files[xml].append(file)
                break
    return dict(grouped_files)