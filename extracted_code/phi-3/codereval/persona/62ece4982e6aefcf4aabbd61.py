def _dictsum(dicts):

    result = {}

    for d in dicts:

        for key, value in d.items():

            result[key] = result.get(key, 0) + value

    return result