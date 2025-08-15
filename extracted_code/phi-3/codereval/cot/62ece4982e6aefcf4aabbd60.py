def size_to_bytes(size: str) -> int:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "E": 1024**6, "Z": 1024**7, "Y": 1024**8}
    size = size.upper()
    if not size[-1].isalpha():
        return -1
    number, unit = float(size[:-1]), size[-1]
    if unit not in units:
        return -1
    return int(number * units[unit])