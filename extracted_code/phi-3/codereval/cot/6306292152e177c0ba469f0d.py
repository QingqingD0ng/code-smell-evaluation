import json

def identify_request(request: dict) -> bool:
    try:
        body = json.loads(request.get('body', '{}'))
        return 'events' in body
    except json.JSONDecodeError:
        return False