class MyDict:
    def __init__(self):
        self.items = []

    def __setitem__(self, key, value):
        self.items.append((key, value))
        self.items.sort(key=lambda item: self.items.index(item), reverse=True)

    def popitem(self):
        if self.items:
            key, value = self.items.pop()
            return key, value
        raise KeyError("popitem(): dictionary is empty")