import json
from urllib.parse import urlparse, parse_qs

# Constants for keys to avoid magic strings
REL_HCAROD ='rel'
HCAROD_HCAROD = 'hcard'
FRAGMENT_PARAM = 'fragment'

def parse_diaspora_webfinger(document: str) -> dict:
    """
    Parses a webfinger document in JSON format and extracts the href value of the hcard_url field.
    Returns a dictionary with the hcard_url and fragment (if present) from the parsed URL.
    """
    try:
        # Parse the JSON document
        webfinger_data = json.loads(document)
    except json.JSONDecodeError as e:
        # Raise an error if JSON parsing fails
        raise ValueError(f'Failed to parse JSON document: {e}')

    # Extract the hcard_url from the webfinger_data
    hcard_url = webfinger_data.get(REL_HCAROD, {}).get(HCAROD_HCAROD, {}).get('href')

    # If no href is found, return an empty dictionary
    if not hcard_url:
        return {}

    # Parse the URL and extract the fragment
    parsed_url = urlparse(hcard_url)
    fragment = parsed_url.fragment

    # Return a dictionary with the extracted hcard_url and fragment
    return {
        'hcard_url': hcard_url,
        'fragment': fragment
    }