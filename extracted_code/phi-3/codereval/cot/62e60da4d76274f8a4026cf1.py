class DataStore:
    def __init__(self, index):
        self.index = index
        self.data = {}

    def values(self, *keys):
        return [self.data[key] for key in keys if key in self.data]