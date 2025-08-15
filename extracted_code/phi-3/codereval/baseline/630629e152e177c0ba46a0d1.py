import requests
from typing import Optional

def try_retrieve_webfinger_document(handle: str) -> Optional[str]:
    url = f"https://webfinger.net/{handle}?_format=application/jrd+json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.text
    except requests.RequestException:
        return None