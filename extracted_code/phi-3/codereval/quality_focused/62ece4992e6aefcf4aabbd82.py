import re

def get_pattern(pattern, strip=True):
    if isinstance(pattern, str):
        pattern = pattern.strip() if strip else pattern
        return re.compile(pattern)
    return pattern