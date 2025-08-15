import xml.etree.ElementTree as ET

def xml_children_as_dict(node):
    result = {}
    for child in node:
        result[child.tag] = child.text
    return result