class DictionaryUtils:
    @staticmethod
    def insert_nested_value(dictionary, value, key, *sub_keys):
        current_dict = dictionary
        for sub_key in sub_keys[:-1]:
            current_dict = current_dict.setdefault(sub_key, {})
        current_dict[sub_keys[-1]] = value

# Usage example
nested_dict = {}
DictionaryUtils.insert_nested_value(nested_dict, 'val', 'key1', 'key2')
print(nested_dict)  # Output: {'key1': {'key2': 'val'}}