import xml.etree.ElementTree as ET

def xml_children_as_dict(node):
    result = {}
    for child in node:
        result[child.tag] = child.text
    return result

xml_str = """
<root>
    <child1>Text1</child1>
    <child2>Text2</child2>
</root>
"""

root = ET.fromstring(xml_str)
children_dict = xml_children_as_dict(root)
print(children_dict)