from collections import defaultdict

def _dictsum(dicts):
    summed_dict = defaultdict(int)
    for d in dicts:
        for key, value in d.items():
            summed_dict[key] += value
    return dict(summed_dict)