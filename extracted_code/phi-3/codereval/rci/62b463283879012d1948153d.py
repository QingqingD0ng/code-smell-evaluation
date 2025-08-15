import xml.etree.ElementTree as ET
from typing import Optional, List

def match_pubdate(node: ET.Element, pubdate_xpaths: List[str]) -> Optional[str]:
    if len(pubdate_xpaths) == 1:
        xpath = pubdate_xpaths[0]
        try:
            return node.find(xpath).text
        except ET.ParseError:
            return None
    
    for xpath in pubdate_xpaths:
        try:
            pubdate_element = node.find(xpath)
            if pubdate_element is not None:
                return pubdate_element.text
        except ET.ParseError:
            continue
    
    return None

# Example usage:
xml_data = '<root><pubdate>2023-04-01</pubdate></root>'
root = ET.fromstring(xml_data)
pubdate_xpaths = ['pubdate']
print(match_pubdate(root, pubdate_xpaths))