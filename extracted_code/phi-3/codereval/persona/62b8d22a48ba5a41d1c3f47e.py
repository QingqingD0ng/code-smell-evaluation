class CustomDict(dict):
    def setdefault(self, key, default=None):
        return self.setdefault(key, default)