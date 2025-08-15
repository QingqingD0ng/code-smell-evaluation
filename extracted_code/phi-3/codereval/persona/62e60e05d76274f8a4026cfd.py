class QualityExpert:
    def __init__(self, data):
        self.data = data

    def index(self, key):
        """Returns the key in the form of int."""
        return self.data.index(key)