import urllib.request

from lxml import etree

import pandas as pd

import re


def validate_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  #...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, url) is not None


def task_func(url):
    if not validate_url(url):
        raise ValueError("Invalid URL")

    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = etree.fromstring(xml_data)
        items = []
        for item in root.findall('.//item'):
            item_data = {}
            for child in item:
                item_data[child.tag] = child.text.strip() if child.text else None
            items.append(item_data)
        df = pd.DataFrame(items)
        # Optional: Convert data types for better consistency
        df = df.apply(pd.to_numeric, errors='ignore')
        # Optional: Remove rows with null values
        df = df.dropna()
        return df
    except urllib.error.HTTPError as e:
        raise ValueError("HTTP Error: " + str(e.code))
    except urllib.error.URLError as e:
        raise ValueError("URL Error: " + str(e.reason))
    except etree.XMLSyntaxError