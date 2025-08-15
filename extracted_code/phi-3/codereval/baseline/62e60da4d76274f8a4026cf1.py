class DataHandler:
    def __init__(self, data):
        self.data = data
        self.index = {}

    def values(self, *keys):
        return [self.data[key] for key in keys if key in self.index]