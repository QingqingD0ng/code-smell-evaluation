class RandomDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def popitem(self):
        return self.__choice()

    def __choice(self):
        key, value = random.choice(list(self.items()))
        self.pop(key)
        return key, value

# Usage example:
# random_dict = RandomDict(a=1, b=2, c=3)
# item = random_dict.popitem()
# print(item)  # This will print a random key-value pair, e.g., ('b', 2)