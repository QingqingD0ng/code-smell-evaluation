import json
import re
import pandas as pd

def task_func(json_str):
    if not json_str.strip():
        return pd.DataFrame()

    data = json.loads(json_str)

    def double_value(value):
        if isinstance(value, (int, float)):
            return float(value) * 2
        elif isinstance(value, str):
            numbers = re.findall(r'-?\d+(?:\.\d+)?', value)
            if numbers:
                return [float(num) * 2 for num in numbers]
            return value
        elif isinstance(value, list):
            return [double_value(item) for item in value]
        return value

    normalized_data = {key: double_value(value) for key, value in data.items()}

    if isinstance(next(iter(normalized_data.values())), list):
        df = pd.DataFrame(normalized_data)
    else:
        df = pd.DataFrame([normalized_data])

    return df