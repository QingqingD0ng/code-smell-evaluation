import re

def validate_key(key):
    pattern = r'^[A-Za-z0-9_-]{4,16}$'
    return bool(re.match(pattern, key))