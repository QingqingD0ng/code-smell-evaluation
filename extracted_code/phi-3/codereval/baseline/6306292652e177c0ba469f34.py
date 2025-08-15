import requests
from typing import Optional

def fetch_content_type(url: str) -> Optional[str]:
    try:
        response = requests.head(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            return response.headers.get('Content-Type', None)
    except requests.RequestException:
        pass
    return None