def normalize_cmd(cmd: tuple[str,...]) -> tuple[str,...]:
    normalized_cmd = []
    for part in cmd:
        if'' in part:
            normalized_cmd.append(f'"{part}"')
        else:
            normalized_cmd.append(part)
    return tuple(normalized_cmd)