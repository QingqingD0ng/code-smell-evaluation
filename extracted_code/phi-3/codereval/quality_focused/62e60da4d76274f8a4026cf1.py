class DataStore:
    def __init__(self):
        self.index = {}  # Initialize the index dictionary

    def add_entry(self, key, value):
        self.index[key] = value

    def values(self, *keys):
        return [self.index[key] for key in keys if key in self.index]

# Example usage:
data_store = DataStore()
data_store.add_entry('apple', 1.2)
data_store.add_entry('banana', 0.5)
data_store.add_entry('cherry', 0.8)

print(data_store.values('apple', 'cherry'))  # Output: [1.2, 0.8]