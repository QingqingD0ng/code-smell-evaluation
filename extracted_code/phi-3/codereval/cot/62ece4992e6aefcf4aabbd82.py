import re

def get_pattern(pattern, strip=True):
    if isinstance(pattern, str):
        if strip:
            pattern = pattern.strip()
        return re.compile(pattern)
    elif isinstance(pattern, re.Pattern):
        return pattern
    else:
        raise ValueError("Input must be a string or an existing regex pattern object")