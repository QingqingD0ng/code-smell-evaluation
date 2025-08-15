import urllib.request
from lxml import etree
import pandas as pd

def task_func(url):
    with urllib.request.urlopen(url) as response:
        xml_content = response.read()
        xml_tree = etree.fromstring(xml_content)
    if xml_tree.tag!= 'items':
        raise ValueError("XML structure does not match expected format.")
    items_data = [{child.tag: child.text for child in item} for item in xml_tree.findall('.//item')]
    df = pd.DataFrame(items_data)
    return df