import json
import xlwt
import os
import pandas as pd

def task_func(json_str, filename, sheet_name="sheet1"):
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError("json_str must be a string, bytes, or bytearray.")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string.")
    
    if not data:
        data = pd.DataFrame()
    else:
        data = pd.DataFrame(data)
    
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    
    for col_num, col_name in enumerate(data.columns):
        sheet.write(0, col_num, col_name)
        for row_num in range(1, len(data) + 1):
            sheet.write(row_num, col_num, data.iloc[row_num - 1, col_num])
    
    workbook.save(filename)
    
    return os.path.abspath(filename)