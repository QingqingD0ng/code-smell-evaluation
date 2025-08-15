import re
from typing import NamedTuple

class SizeUnits(NamedTuple):
    unit: str
    factor: int

def parse_size_str(size_str: str) -> tuple[float, str]:
    size_str = size_str.strip().upper()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', size_str)
    
    if not match or len(match.groups())!= 2:
        raise ValueError(f"Invalid size format: {size_str}")
    
    number, unit = match.groups()
    if not unit:
        return float(number), ''
    
    unit = unit[0]
    if unit == 'K':
        return float(number) * 1024, 'KB'
    elif unit == 'M':
        return float(number) * 1024**2, 'MB'
    elif unit == 'G':
        return float(number) * 1024**3, 'GB'
    elif unit == 'T':
        return float(number) * 1024**4, 'TB'
    else:
        raise ValueError(f"Invalid size unit: {unit}")

def size_to_bytes(size_str: str) -> int:
    number, unit = parse_size_str(size_str)
    unit_factor = {
        'KB': SizeUnits('KB', 1024),
        'MB': SizeUnits('MB', 1024**2),
        'GB': SizeUnits('GB', 1024**3),
        'TB': SizeUnits('TB', 1024**4),
    }
    return int(number * unit_factor[unit].factor)