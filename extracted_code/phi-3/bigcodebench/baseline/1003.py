import urllib.request

from lxml import etree

import pandas as pd


def task_func(url):
    response = urllib.request.urlopen(url)
    xml_data = response.read()
    root = etree.fromstring(xml_data)
    items = []
    for item in root.findall('.//item'):
        item_data = {}
        for child in item:
            item_data[child.tag] = child.text
        items.append(item_data)
    df = pd.DataFrame(items)
    return df