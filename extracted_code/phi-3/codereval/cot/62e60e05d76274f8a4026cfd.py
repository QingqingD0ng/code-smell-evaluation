class Indexer:
    def __init__(self):
        self.index = {}

    def index(self, key):
        if key not in self.index:
            self.index[key] = len(self.index)
        return self.index[key]