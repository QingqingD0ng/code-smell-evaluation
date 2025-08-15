import re


def validate_key(key):

    pattern = r'^[a-zA-Z0-9_]{1,20}$'  # Example pattern: alphanumeric and underscore, 1-20 chars

    return bool(re.match(pattern, key))