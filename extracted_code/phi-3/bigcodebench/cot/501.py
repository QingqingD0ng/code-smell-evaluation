import xlwt
import os
import pandas as pd
import json

def task_func(json_str, filename, sheet_name="sheet1"):
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError("json_str must be a string, bytes or bytearray")
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")
    df = pd.json_normalize(data)
    if df.empty:
        df = pd.DataFrame()
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(sheet_name)
    for idx, row in enumerate(df.iterrows()):
        for col, value in enumerate(row[1]):
            worksheet.write(idx, col, value)
    file_path = os.path.abspath(filename)
    workbook.save(file_path)
    return file_path