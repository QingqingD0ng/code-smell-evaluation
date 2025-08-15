import json
import re
import pandas as pd

def task_func(json_str):
    try:
        data_dict = json.loads(json_str)
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                data_dict[key] = value * 2
            elif isinstance(value, str):
                match = re.match(r'^(\d+(\.\d+)?)$', value)
                if match:
                    data_dict[key] = float(match.group(1)) * 2
            elif isinstance(value, list):
                data_dict[key] = [float(item) * 2 if item.replace('.', '', 1).isdigit() else item for item in value]
            else:
                pass  # Ignore non-numeric values or nested structures
        df = pd.DataFrame([data_dict])
        return df
    except json.JSONDecodeError:
        return pd.DataFrame()