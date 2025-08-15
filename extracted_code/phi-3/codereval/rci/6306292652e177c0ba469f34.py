import requests
from typing import Optional

def get_response_content_type(url: str) -> Optional[str]:
    """Return the Content-Type of a resource using HEAD request."""
    try:
        response = requests.head(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            return response.headers.get('Content-Type', None)
    except requests.RequestException:
        pass
    return None