import json
import re
import pandas as pd

def task_func(json_str):
    data = json.loads(json_str)
    if not data:
        return pd.DataFrame()

    def double_value(value):
        if isinstance(value, (int, float)):
            return value * 2.0
        elif isinstance(value, list):
            return [double_value(v) for v in value]
        elif isinstance(value, str):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            if numbers:
                return sum(double_value(float(num)) for num in numbers)
            else:
                return value
        else:
            return value

    normalized_data = {k: double_value(v) for k, v in data.items()}

    def to_list(d):
        max_length = max((len(v) if isinstance(v, list) else 1) for v in d.values())
        return {k: (v if isinstance(v, list) else [v] * max_length) for k, v in d.items()}

    list_data = to_list(normalized_data)
    df = pd.DataFrame(list_data)
    return df