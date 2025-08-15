import json
import pandas as pd
import re

def task_func(json_str):
    data_dict = json.loads(json_str)
    normalized_dict = {}

    def double_value(value):
        if isinstance(value, list):
            return [double_value(item) for item in value]
        elif isinstance(value, float) or isinstance(value, int):
            return value * 2
        elif isinstance(value, str) and re.match(r'^-?\d+(?:\.\d+)?$', value):
            return float(value)
        else:
            return value

    for key, value in data_dict.items():
        normalized_dict[key] = double_value(value)

    return pd.DataFrame.from_dict(normalized_dict, orient='index').astype(float)