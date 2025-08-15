def normalize_cmd(cmd: tuple[str,...]) -> tuple[str,...]:
    normalized_cmd = tuple("\\".join(part.split("\\")) for part in cmd)
    return normalized_cmd