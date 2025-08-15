import xml.etree.ElementTree as ET

def match_pubdate(node, pubdate_xpaths):
    for xpath in pubdate_xpaths:
        try:
            pubdate_element = node.find(xpath)
            if pubdate_element is not None:
                return pubdate_element.text
        except ET.ParseError:
            continue
    return None

# Example usage:
# xml_data = '<root><pubdate>2023-04-01</pubdate></root>'
# root = ET.fromstring(xml_data)
# pubdate_xpaths = ['pubdate']
# print(match_pubdate(root, pubdate_xpaths))