class RandomPopDict:
    def __init__(self):
        self.items = {}

    def __choice(self):
        # Randomly select a key from the dictionary
        return random.choice(list(self.items.keys()))

    def popitem(self):
        if not self.items:
            raise KeyError("popitem(): dictionary is empty")
        key = self.__choice()
        return key, self.items.pop(key)