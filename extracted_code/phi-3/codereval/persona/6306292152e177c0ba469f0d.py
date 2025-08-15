import json

def identify_request(request: dict) -> bool:
    return 'events' in request.get('body', {})