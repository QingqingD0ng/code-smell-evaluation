def is_ipv4(target):

    parts = target.split('.')

    if len(parts)!= 4:

        return False

    for part in parts:

        if not part.isdigit() or not 0 <= int(part) <= 255:

            return False

    return True