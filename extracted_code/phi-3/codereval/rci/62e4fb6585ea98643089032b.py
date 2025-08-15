def parse_version(s: str) -> tuple[int,...]:
    if not isinstance(s, str):
        raise AttributeError(f"Input must be a string. Got {type(s)} instead.")
    parts = s.split('.')
    if not all(part.isdigit() for part in parts):
        raise ValueError(f"Input string '{s}' must contain only integers separated by '.'.")
    return tuple(int(part) for part in parts)