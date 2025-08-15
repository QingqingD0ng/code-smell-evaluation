class CustomDict:
    def __init__(self):
        self._dict = {}

    def __setitem__(self, key, value):
        self._dict[key] = value

    def popitem(self):
        return self._dict.popitem()