dict_insert = lambda dic, val, key, *keys: set_nested_value(dic, key.split('.'), val)

set_nested_value = lambda dic, keys, val: reduce(lambda d, k: {k: d[k] if k in d else {}}, keys[:-1], {k: d[k] if k in d else None for k in keys})[keys[-1]] = val