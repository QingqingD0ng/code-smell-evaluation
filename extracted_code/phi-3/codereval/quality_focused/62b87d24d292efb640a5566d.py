def plus_or_dot(pieces):
    return '+' if pieces and pieces[-1].startswith('-') and '+' in pieces[-1] else '.'