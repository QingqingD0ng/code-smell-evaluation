import requests
from typing import Optional

def try_retrieve_webfinger_document(handle: str) -> Optional[str]:
    url = f"http://hostname/.well-known/webfinger?resource=acct:{handle}@hostname"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None