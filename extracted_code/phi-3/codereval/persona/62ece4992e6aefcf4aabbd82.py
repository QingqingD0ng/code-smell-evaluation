import re

def get_pattern(pattern, strip=True):
    if isinstance(pattern, str):
        if strip:
            pattern = pattern.strip()
        regex_pattern = re.compile(pattern)
    elif isinstance(pattern, re.Pattern):
        regex_pattern = pattern
    else:
        raise ValueError("Input must be a string or a regex pattern object")
    return regex_pattern