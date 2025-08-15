def parse_version(s: str) -> tuple[int,...]:
    return tuple(map(int, s.split('.')))