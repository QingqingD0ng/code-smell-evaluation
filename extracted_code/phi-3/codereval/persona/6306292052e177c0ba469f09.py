import json
from xml.etree import ElementTree as ET
from typing import Union, Any

RequestType = Union[str, bytes]

def identify_request(request: RequestType) -> bool:
    if isinstance(request, str):
        try:
            data = json.loads(request)
            return 'events' in data
        except json.JSONDecodeError:
            pass
    elif isinstance(request, bytes):
        try:
            xml_data = ET.fromstring(request.decode())
            return xml_data.tag == 'Magic_ENV_TAG'
        except ET.ParseError:
            pass
    return False