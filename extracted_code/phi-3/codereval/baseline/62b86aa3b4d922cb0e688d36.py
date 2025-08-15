import re

def validate_key(key):
    return re.match(r'^[a-zA-Z0-9_]+$', key) is not None

def validate_value(value):
    return isinstance(value, (int, float, str))

def _validate_labels(labels):
    for key, value in labels.items():
        if not validate_key(key):
            raise ValueError(f"Invalid key: {key}")
        if not validate_value(value):
            raise ValueError(f"Invalid value for key '{key}': {value}")