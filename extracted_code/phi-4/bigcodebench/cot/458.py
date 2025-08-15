import json
import re
import pandas as pd

def task_func(json_str):
    data = json.loads(json_str)
    
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [double_value(v) for v in value]
        elif isinstance(value, (int, float)):
            data[key] = double_value(value)
        elif isinstance(value, str):
            data[key] = double_value(value)
    
    if data:
        max_len = max(len(v) if isinstance(v, list) else 1 for v in data.values())
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = value + [value[-1]] * (max_len - len(value))
            else:
                data[key] = [value] * max_len

    return pd.DataFrame(data)

def double_value(value):
    if isinstance(value, (int, float)):
        return value * 2
    if isinstance(value, str) and re.match(r'^-?\d+(\.\d+)?$', value):
        return float(value) * 2
    return value