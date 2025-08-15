class MyClass:
    def __init__(self):
        self.__keys = ['key1', 'key2', 'key3']

    def get_keys(self):
        return self.__keys.copy()

    def add_key(self, key):
        self.__keys.append(key)

    def remove_key(self, key):
        if key in self.__keys:
            self.__keys.remove(key)

    def get_keys_as_set(self):
        return set(self.__keys)

    def get_keys_as_list(self):
        return self.__keys.copy()