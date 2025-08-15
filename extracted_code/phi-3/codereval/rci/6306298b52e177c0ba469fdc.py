import xml.etree.ElementTree as ET
from typing import Union, Dict, Any

def xml_children_as_dict(node: ET.Element) -> Dict[str, Union[str, Dict[str, Any]]]:
    if not isinstance(node, ET.Element):
        raise ValueError("Input must be an XML ElementTree node")

    result = {}
    for child in node:
        result[child.tag] = {}
        if child.text and child.text.strip():
            result[child.tag]['text'] = child.text.strip()
        if child.attrib:
            result[child.tag]['attributes'] = child.attrib
        for subchild in child:
            if subchild.tag not in result[child.tag]:
                result[child.tag].update(xml_children_as_dict(subchild))
    return result

# Example usage:
xml_str = '''<root>
    <child1>text1</child1>
    <child2 attributes="value">text2</child2>
    <child3>
        <nested>nested text</nested>
    </child3>
</root>'''
root = ET.fromstring(xml_str)
children_dict = xml_children_as_dict(root)
print(children_dict)