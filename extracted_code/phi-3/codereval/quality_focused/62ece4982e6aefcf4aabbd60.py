def size_to_bytes(size: str) -> int:

    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "E": 1024**6}

    number, unit = int(size[:-1]), size[-1]

    return number * units[unit]