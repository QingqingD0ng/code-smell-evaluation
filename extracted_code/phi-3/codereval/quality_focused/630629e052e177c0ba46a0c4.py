import json
from typing import Dict

def parse_diaspora_webfinger(document: str) -> Dict:
    data = json.loads(document)
    webfinger_data = data.get('links', [])
    href_values = [link.get('href') for link in webfinger_data if link.get('rel') == 'alternate']
    return {'hcard_url': href_values[0] if href_values else None}