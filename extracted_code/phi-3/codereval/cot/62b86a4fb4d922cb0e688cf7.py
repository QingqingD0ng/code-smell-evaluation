import re

def validate_key(key):
    pattern = r'^[a-zA-Z0-9_-]+$'
    return re.match(pattern, key) is not None

key_to_validate = "valid-key123"
print(f"The key '{key_to_validate}' is {'valid' if validate_key(key_to_validate) else 'invalid'}")