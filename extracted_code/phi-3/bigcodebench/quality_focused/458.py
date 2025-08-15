import json
import re
import pandas as pd

def task_func(json_str):
    if not json_str:
        return pd.DataFrame()

    data_dict = json.loads(json_str)
    normalized_dict = {}

    for key, value in data_dict.items():
        if isinstance(value, list):
            normalized_dict[key] = [float(x) * 2 if isinstance(x, (int, float)) else x for x in value]
        elif isinstance(value, str):
            match = re.match(r'^(\d+(\.\d+)?)$', value)
            normalized_dict[key] = float(match.group(1)) * 2 if match else value
        else:
            normalized_dict[key] = value * 2 if isinstance(value, (int, float)) else value

    return pd.DataFrame(normalized_dict)