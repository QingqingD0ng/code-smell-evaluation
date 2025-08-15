class RandomPickableDict:
    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)

    def popitem(self):
        if self._data:
            return self.__choice()
        else:
            raise KeyError("popitem(): dictionary is empty")

    def __choice(self):
        key = next(iter(self._data))
        return key, self._data.pop(key)

# Example usage:
# my_dict = RandomPickableDict({'a': 1, 'b': 2, 'c': 3})
# random_key, random_value = my_dict.popitem()
# print(random_key, random_value)