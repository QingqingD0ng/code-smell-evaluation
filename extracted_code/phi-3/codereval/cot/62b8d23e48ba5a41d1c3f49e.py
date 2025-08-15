import random

class RandomDict:
    def __init__(self):
        self._dict = {}

    def __choice(self):
        return random.choice(list(self._dict.keys()))

    def popitem(self):
        key = self.__choice()
        return key, self._dict.pop(key)