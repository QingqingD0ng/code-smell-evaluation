class MyClass:
    def __init__(self):
        self._data = {}

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)