def difference(d1, d2, level=-1):

    def filter_dict(d, keys_to_exclude):

        return {k: v for k, v in d.items() if k not in keys_to_exclude}


    if level == 0:

        return filter_dict(d1, d2.keys())

    elif level > 0:

        d2_filtered = filter_dict(d2, d1.keys())

        return difference(d1, d2_filtered, level - 1)

    else:

        return filter_dict(d1, d2.keys())