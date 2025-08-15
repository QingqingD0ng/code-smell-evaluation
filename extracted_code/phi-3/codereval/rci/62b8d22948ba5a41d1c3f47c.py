class MyDict:
    def __init__(self):
        self.__data = {}

    def get(self, key: str, default: str = None) -> str:
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        return self.__data.get(key, default)

    def set(self, key: str, value: str):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Key and value must be strings")
        self.__data[key] = value

    def update(self, updates: dict):
        if not isinstance(updates, dict):
            raise TypeError("Updates must be a dictionary")
        for k, v in updates.items():
            self.set(k, v)

# Example usage:
# my_dict = MyDict()
# my_dict.set('key1', 'value1')
# print(my_dict.get('key1'))
# my_dict.update({'key2': 'value2', 'key3': 'value3'})
# print(my_dict.get('key2'))