class CustomDict:
    def __init__(self):
        self.items = []

    def __setitem__(self, key, value):
        self.__delitem__(key)
        self.items.append((key, value))
        self.items.sort(key=lambda item: self.items.index(item), reverse=True)

    def __getitem__(self, key):
        for k, v in self.items:
            if k == key:
                return v
        raise KeyError(key)

    def __delitem__(self, key):
        self.items = [(k, v) for k, v in self.items if k!= key]

    def popitem(self):
        if not self.items:
            raise KeyError('popitem(): dictionary is empty')
        return self.items.pop()

# Example usage:
# d = CustomDict()
# d['a'] = 1
# d['b'] = 2
# d['c'] = 3
# print(d.popitem())  # Output: ('c', 3)