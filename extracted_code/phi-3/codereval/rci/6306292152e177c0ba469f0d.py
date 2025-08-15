import json

import logging


def identify_request(request: dict) -> bool:
    try:
        request_data = json.loads(request)
        return 'events' in request_data
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in the request: {e}")
        return False