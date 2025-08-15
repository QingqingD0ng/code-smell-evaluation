def find_metaclass(bases, explicit_mc=None) -> type:
    if explicit_mc is not None:
        return explicit_mc
    metaclass = object
    for base in reversed(bases):
        if base is type:
            metaclass = base
            break
    return metaclass