def minimalBases(classes):
    minimal_bases = []
    for cls in classes:
        if not any(issubclass(c, cls) for c in classes if c!= cls):
            minimal_bases.append(cls)
    return minimal_bases