import os

import xml.etree.ElementTree as ET


def _group_files_by_xml_filename(files):
	groups = {}
	for file in files:
		root = ET.parse(file).getroot()
		filename = os.path.splitext(root.find('filename').text)[0]
		groups.setdefault(filename, []).append(file)
	return groups


def _explore_folder(folder):
	for root, _, files in os.walk(folder):
		grouped_files = _group_files_by_xml_filename(files)
		for group, files in grouped_files.items():
			print(f"Group: {group}, Files: {files}")


# Example usage:

# _explore_folder('/path/to/your/xml/folder')