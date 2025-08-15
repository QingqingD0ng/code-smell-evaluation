import json
from typing import Any, Dict

RequestType = Dict[str, Any]

def identify_request(request: RequestType) -> bool:
    if 'body' in request and isinstance(request['body'], str):
        try:
            body_data = json.loads(request['body'])
            if isinstance(body_data, list) and all(isinstance(event, dict) for event in body_data):
                return True
        except json.JSONDecodeError:
            pass
    return False