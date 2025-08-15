def pop(self, key, default=__marker):
    try:
        return self[key]
    except KeyError:
        if default is not __marker:
            return default
        raise