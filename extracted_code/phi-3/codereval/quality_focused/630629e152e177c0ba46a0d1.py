import requests

from typing import Optional


def try_retrieve_webfinger_document(handle: str) -> Optional[str]:

    url = f"http://example.com/.well-known/webfinger?resource=acct:{handle}@example.com"

    try:

        response = requests.get(url)

        response.raise_for_status()  # Raises HTTPError for bad responses

        return response.text

    except requests.RequestException:

        return None