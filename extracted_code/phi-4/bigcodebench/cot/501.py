import xlwt
import os
import pandas as pd
import json

def task_func(json_str, filename, sheet_name="sheet1"):
    if isinstance(json_str, (bytes, bytearray)):
        json_str = json_str.decode('utf-8')
    if not isinstance(json_str, str):
        raise TypeError("json_str must be a string, bytes, or bytearray.")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string.")

    if isinstance(data, list) and len(data) == 0:
        data = []

    if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()

    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(sheet_name)

    if not df.empty:
        for r, row in enumerate(df.to_dict(orient='records')):
            for c, (col, val) in enumerate(row.items()):
                worksheet.write(r, c, val)
    else:
        worksheet.write(0, 0, '')

    workbook.save(filename)
    return os.path.abspath(filename)