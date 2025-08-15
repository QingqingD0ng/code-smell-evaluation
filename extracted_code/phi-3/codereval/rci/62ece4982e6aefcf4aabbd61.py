def sum_values_with_same_keys(dicts):

    if not all(isinstance(d, dict) for d in dicts):

        raise ValueError("All elements must be dictionaries")

    result = {}

    for d in dicts:

        for k, v in d.items():

            result[k] = result.get(k, 0) + v

    return result