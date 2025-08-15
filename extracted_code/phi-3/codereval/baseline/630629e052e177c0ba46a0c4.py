import json
from urllib.parse import urlparse, parse_qs

def parse_diaspora_webfinger(document: str) -> dict:
    webfinger_data = json.loads(document)
    hcard_url = webfinger_data.get('rel', {}).get('hcard', {}).get('href')
    
    if hcard_url:
        parsed_url = urlparse(hcard_url)
        query_params = parse_qs(parsed_url.query)
        # Assuming we want to extract the fragment if present
        fragment = query_params.get('fragment', [None])[0]
        return {'hcard_url': hcard_url, 'fragment': fragment}
    else:
        return {}