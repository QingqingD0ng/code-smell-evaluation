import json
import pandas as pd
import re

def task_func(json_str):
    if not json_str:
        return pd.DataFrame()

    data_dict = json.loads(json_str)

    for key, value in data_dict.items():
        if isinstance(value, list):
            data_dict[key] = [float(x) * 2 if isinstance(x, (int, float)) else x for x in value]
        elif isinstance(value, str) and re.match(r'^-?\d+(\.\d+)?$', value):
            data_dict[key] = float(value) * 2

    return pd.DataFrame(data_dict).astype(float)