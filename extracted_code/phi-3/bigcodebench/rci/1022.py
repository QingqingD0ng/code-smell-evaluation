import pandas as pd
import os
from datetime import datetime
from pandas.errors import EmptyDataError

def read_csv_file(csv_file_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"File Not Found: {csv_file_path}")
    
    if not os.access(csv_file_path, os.R_OK):
        raise FileNotFoundError(f"File Not Accessible: {csv_file_path}")
    
    try:
        df = pd.read_csv(csv_file_path)
    except EmptyDataError:
        return pd.DataFrame()
    
    return df

def validate_column(df: pd.DataFrame, column_name: str) -> None:
    if column_name not in df.columns:
        raise ValueError(f"Column not found: {column_name}")

def convert_dates(df: pd.DataFrame, column_name: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    df[column_name] = pd.to_datetime(df[column_name], format=date_format)
    return df

def filter_by_current_date(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    current_date = datetime.now()
    df_filtered = df[df[column_name] > current_date]
    return df_filtered

def sort_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df_sorted = df.sort_values(by=column_name)
    return df_sorted

def task_func(csv_file_path: str, column_name: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    df = read_csv_file(csv_file_path)
    validate_column(df, column_name)
    df = convert_dates(df, column_name, date_format)
    df = filter_by_current_date(df, column_name)
    df = sort_data(df,