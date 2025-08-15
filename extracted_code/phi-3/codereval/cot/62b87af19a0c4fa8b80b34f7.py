def difference(d1, d2, level=-1):
    result = {}
    for key, value in d1.items():
        if level > 0 and key not in d2:
            continue
        if key not in d2 or (level > 0 and value!= d2[key]):
            result[key] = value
    return result