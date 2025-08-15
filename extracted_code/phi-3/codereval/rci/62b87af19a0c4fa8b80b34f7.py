def difference(d1, d2, level=-1):
    d1_diff = {}
    for k in d1:
        if k not in d2 or d1[k]!= d2.get(k, None):
            d1_diff[k] = d1[k]
    return d1_diff