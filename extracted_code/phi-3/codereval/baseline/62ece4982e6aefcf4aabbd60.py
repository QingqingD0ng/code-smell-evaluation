import re

def size_to_bytes(size: str) -> int:
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    if not re.match(r'^\d+(\.\d+)?[KMGT]*$', size.upper()):
        raise ValueError("Invalid size format")
    number, unit = float(re.sub(r'[^\d.]', '', size)), re.sub(r'[^\d.]', '', size)[-1]
    return int(number * units[unit]) if unit else int(number)