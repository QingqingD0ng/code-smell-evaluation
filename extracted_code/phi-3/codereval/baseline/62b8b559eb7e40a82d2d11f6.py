def determineMetaclass(bases, explicit_mc=None):
    if explicit_mc is not None:
        return explicit_mc
    else:
        metaclass = object
        for base in reversed(bases):
            if isinstance(base, type):
                metaclass = base
                break
        return metaclass