def dict_insert(dic, val, key, *keys):
    for k in keys[:-1]:
        dic = dic.setdefault(k, {})
    dic[keys[-1]] = val