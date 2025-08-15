class MyClass:
    def __init__(self):
        self.data = {}

    def setdefault(self, key, default=None):
        return self.data.setdefault(key, default)