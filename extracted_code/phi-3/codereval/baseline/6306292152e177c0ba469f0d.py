import json

def identify_request(request: dict) -> bool:
    try:
        request_data = json.loads(request)
        return 'events' in request_data
    except json.JSONDecodeError:
        return False