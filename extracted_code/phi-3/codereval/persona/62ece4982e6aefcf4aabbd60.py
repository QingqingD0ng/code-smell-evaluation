def size_to_bytes(size: str) -> int:
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    size = size.upper()
    if size[-1] not in units:
        raise ValueError("Invalid size unit. Use K, M, G, or T.")
    num_bytes = int(size[:-1]) * units[size[-1]]
    return num_bytes