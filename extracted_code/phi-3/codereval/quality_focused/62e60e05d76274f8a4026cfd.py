class DictionaryIndex:
    def __init__(self):
        self.index = {}

    def insert(self, key, value):
        self.index[key] = value

    def search(self, key):
        return self.index.get(key)

# Usage
dict_index = DictionaryIndex()
dict_index.insert(1, "One")
dict_index.insert(2, "Two")

print(dict_index.search(1))  # Output: One
print(dict_index.search(3))  # Output: None, since 3 is not in the dictionary