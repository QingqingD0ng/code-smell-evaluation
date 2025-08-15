import random

class RandomPopDict:
    def __init__(self):
        self.items = {}

    def _get_random_key(self):
        return random.choice(list(self.items.keys()))

    def popitem(self):
        if not self.items:
            return None
        key = self._get_random_key()
        return key, self.items.pop(key)