import json
import re
import pandas as pd

def task_func(json_str):
    if not json_str.strip():
        return pd.DataFrame()

    data = json.loads(json_str)

    def double_value(value):
        if isinstance(value, (int, float)):
            return value * 2
        elif isinstance(value, str):
            match = re.match(r'^-?\d+(\.\d+)?$', value)
            return float(match.group()) * 2 if match else value
        elif isinstance(value, list):
            return [double_value(v) for v in value]
        return value

    normalized_data = {k: double_value(v) for k, v in data.items()}

    if isinstance(normalized_data.get(list(normalized_data.keys())[0]), list):
        max_length = max(len(v) for v in normalized_data.values() if isinstance(v, list))
        for key, value in normalized_data.items():
            if isinstance(value, list):
                normalized_data[key] = value + [value[-1]] * (max_length - len(value))
        df = pd.DataFrame(normalized_data)
    else:
        df = pd.DataFrame([normalized_data])

    return df