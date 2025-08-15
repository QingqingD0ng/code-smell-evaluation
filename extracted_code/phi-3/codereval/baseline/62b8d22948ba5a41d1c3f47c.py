class MyDict:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

# Example usage:
# my_dict = MyDict()
# my_dict.get('key1', 'default_value')