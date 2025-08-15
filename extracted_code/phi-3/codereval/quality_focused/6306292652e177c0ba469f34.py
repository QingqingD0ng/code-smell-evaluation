import requests
from typing import Optional

def fetch_content_type(url: str) -> Optional[str]:
    try:
        response = requests.head(url, headers={'User-Agent': 'Mozilla/5.0'})
        content_type = response.headers.get('Content-Type', None)
        return content_type
    except requests.RequestException:
        return None