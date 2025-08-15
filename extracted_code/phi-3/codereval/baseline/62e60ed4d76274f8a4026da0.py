class MyClass:
    def __init__(self):
        self.__keys = ['key1', 'key2', 'key3']

    def keys(self):
        return self.__keys.copy()