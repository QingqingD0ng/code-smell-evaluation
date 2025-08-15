import urllib.parse
from typing import Tuple

def _parse_image_ref(image_href: str) -> Tuple[str, str, bool]:
    parsed_url = urllib.parse.urlparse(image_href)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"{parsed_url.scheme} is not a supported scheme.")
    image_id = parsed_url.path.lstrip('/')
    netloc = parsed_url.netloc
    use_ssl = parsed_url.scheme == 'https'
    return image_id, netloc, use_ssl