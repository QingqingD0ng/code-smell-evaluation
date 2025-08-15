import json
import re
import pandas as pd

def normalize_value(value):
    if isinstance(value, (int, float)):
        return value * 2
    elif isinstance(value, str):
        match = re.fullmatch(r'^-?\d+(\.\d+)?$', value)
        return float(match.group(0)) * 2 if match else value
    elif isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value

def construct_dataframe_from_dict(data_dict):
    processed_dict = {key: normalize_value(value) for key, value in data_dict.items()}
    df = pd.DataFrame([processed_dict])
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='ignore').astype(float)
    return df

def task_func(json_str):
    try:
        data_dict = json.loads(json_str)
        processed_dict = {key: normalize_value(value) for key, value in data_dict.items()}
        return construct_dataframe_from_dict(processed_dict)
    except json.JSONDecodeError:
        return pd.DataFrame()