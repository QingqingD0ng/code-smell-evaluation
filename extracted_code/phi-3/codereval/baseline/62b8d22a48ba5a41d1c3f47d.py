class Dictionary(dict):
    def pop(self, key, default=object()):
        try:
            value = super().pop(key)
        except KeyError:
            if default is object():
                raise
            return default
        return value