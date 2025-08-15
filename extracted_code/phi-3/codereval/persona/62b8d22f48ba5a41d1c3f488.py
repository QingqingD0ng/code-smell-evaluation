class QualityExpert:
    def popitem(self):
        if self.items:
            key, value = self.items.popitem(last=False)
            return key, value
        else:
            raise KeyError("popitem(): dictionary is empty")

    def __init__(self):
        self.items = {}

    def add_item(self, key, value):
        self.items[key] = value