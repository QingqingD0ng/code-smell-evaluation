class CustomDict:
    def __init__(self):
        self.items = []

    def __setitem__(self, key, value):
        if key in self.items:
            self.items.remove((key, self.items[key]))
        self.items.append((key, value))

    def popitem(self):
        if self.items:
            return self.items.pop()
        raise KeyError("popitem(): dictionary is empty")

# Example usage:
# d = CustomDict()
# d['a'] = 1
# d['b'] = 2
# print(d.popitem())  # Output: ('a', 1)
# print(d.popitem())  # Output: ('b', 2)
# print(d.popitem())  # Raises KeyError: popitem(): dictionary is empty