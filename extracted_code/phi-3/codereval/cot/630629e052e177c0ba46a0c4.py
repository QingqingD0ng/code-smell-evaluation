import json
from typing import Dict, Optional

def parse_diaspora_webfinger(document: str) -> Dict[str, Optional[str]]:
    webfinger_data = json.loads(document)
    hcard_url = webfinger_data.get('hcard_url')
    if not hcard_url:
        return {'hcard_url': None, 'href': None}

    links = webfinger_data.get('links', [])
    href = next((link.get('href') for link in links if link.get('rel') =='self'), None)

    return {'hcard_url': hcard_url, 'href': href}