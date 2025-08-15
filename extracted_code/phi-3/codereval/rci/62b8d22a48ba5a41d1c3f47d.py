class CustomDict(dict):
    def remove_and_get(self, key, default=None):
        try:
            return super().pop(key)
        except KeyError:
            return default

# Example usage:
my_dict = CustomDict({'a': 1, 'b': 2})
value = my_dict.remove_and_get('a')  # Returns 1
value = my_dict.remove_and_get('c')  # Returns None, as 'c' was not in the dictionary

try:
    value = my_dict.remove_and_get('d')
except KeyError as e:
    print(f"KeyError: {e}")  # Prints "KeyError: 'd'"