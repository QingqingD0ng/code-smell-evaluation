import pandas as pd
import os
from datetime import datetime


def task_func(csv_file_path, column_name, date_format="%Y-%m-%d"):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError("CSV file not found.")
    try:
        data = pd.read_csv(csv_file_path)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found.")
        data[column_name] = pd.to_datetime(data[column_name], format=date_format)
        current_date = datetime.now().date()
        filtered_data = data[data[column_name].dt.date >= current_date]
        sorted_data = filtered_data.sort_values(by=column_name)
        return sorted_data
    except pd.errors.EmptyDataError:
        return pd.DataFrame()