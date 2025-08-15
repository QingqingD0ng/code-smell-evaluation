import re

def validate_key(key):
    return bool(re.match(r'^[a-zA-Z0-9_]+$', key))

def validate_value(value):
    return isinstance(value, (int, float, str))

def _validate_labels(labels):
    invalid_entries = []
    for key, value in labels.items():
        if not validate_key(key):
            invalid_entries.append((key, "Invalid key"))
        if not validate_value(value):
            invalid_entries.append((key, "Invalid value"))
    return invalid_entries