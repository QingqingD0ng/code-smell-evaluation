import json

from typing import Dict


def parse_diaspora_webfinger(document: str) -> Dict:

    webfinger_data = json.loads(document)

    hcard_url = None


    for link in webfinger_data.get("links", []):

        if link.get("type") == ["text/html", "application/xhtml+xml"]:

            hcard_url = link.get("href")

            break


    return {"hcard_url": hcard_url}