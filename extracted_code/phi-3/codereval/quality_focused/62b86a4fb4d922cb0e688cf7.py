import re

def validate_key(key):
    # Define a regular expression pattern for valid keys
    pattern = r'^[a-zA-Z0-9_-]{3,16}$'
    
    # Check if the key matches the pattern
    if re.match(pattern, key):
        return True
    else:
        return False