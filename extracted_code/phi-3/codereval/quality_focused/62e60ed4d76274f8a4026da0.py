class MyClass:
    def __init__(self):
        self._keys = ['key1', 'key2', 'key3']  # Example keys, replace with actual keys initialization

    def keys(self):
        return list(self._keys)

# Usage
my_instance = MyClass()
print(my_instance.keys())