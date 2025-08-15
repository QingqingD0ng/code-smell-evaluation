class DictWithPop(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pop(self, key, default=object()):
        if key in self:
            return super().pop(key)
        if default is object():
            raise KeyError(key)
        return default