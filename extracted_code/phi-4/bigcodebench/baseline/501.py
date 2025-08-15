import xlwt
import os
import pandas as pd
import json

def task_func(json_str, filename, sheet_name="sheet1"):
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError("json_str must be a string, bytes, or bytearray")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON")
    
    if not isinstance(data, list):
        raise ValueError("JSON must represent a list")
    
    df = pd.DataFrame(data)
    
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    
    if df.empty:
        workbook.save(filename)
    else:
        for r, row in enumerate(df.to_dict(orient='records'), start=0):
            for c, (col, value) in enumerate(row.items(), start=0):
                sheet.write(r, c, value)
        
        workbook.save(filename)
    
    return os.path.abspath(filename)