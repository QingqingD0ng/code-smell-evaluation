import requests
from typing import Optional

def try_retrieve_webfinger_document(handle: str) -> Optional[str]:
    url = f"https://webfinger.net/{handle}?_format=application/jrd+json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")  # Log specific HTTP errors
        return None
    except requests.RequestException as err:
        logging.error(f"An error occurred: {err}")  # Log other request-related errors
        return None