import xml.etree.ElementTree as ET

def xml_children_as_dict(node):
    return {child.tag: child.text for child in node}

# Example usage:
xml_string = '<xml><tag1>value1</tag1><tag2>value2</tag2></xml>'
root = ET.fromstring(xml_string)
print(xml_children_as_dict(root))