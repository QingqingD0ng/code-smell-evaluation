import json
import re
import pandas as pd

def task_func(json_str, multiplier=2.0):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    def double_value(value):
        if isinstance(value, (int, float)):
            return value * multiplier
        elif isinstance(value, list):
            return double_list(value)
        elif isinstance(value, str):
            return double_string(value)
        else:
            return value

    def double_list(lst):
        return [double_value(v) for v in lst]

    def double_string(s):
        numbers = re.findall(r"[-+]?\d*\.?\d+", s)
        return sum(double_numeric(float(num)) for num in numbers) if numbers else s

    def double_numeric(value):
        return value * multiplier

    normalized_data = {k: double_value(v) for k, v in data.items()}

    def to_list(d):
        max_length = max((len(v) if isinstance(v, list) else 1) for v in d.values())
        return {k: (v if isinstance(v, list) else [v] * max_length) for k, v in d.items()}

    list_data = to_list(normalized_data)
    df = pd.DataFrame(list_data)
    return df