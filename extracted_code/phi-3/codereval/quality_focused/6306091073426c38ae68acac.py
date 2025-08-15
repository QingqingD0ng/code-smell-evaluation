def dict_insert(dic, val, key, *keys):
    for k in keys[:-1]:
        dic = dic.setdefault(k, {})
    dic[keys[-1]] = val

# Example usage:
my_dict = {}
dict_insert(my_dict, 'val', 'key1', 'key2')
print(my_dict)  # Output: {'key1': {'key2': 'val'}}