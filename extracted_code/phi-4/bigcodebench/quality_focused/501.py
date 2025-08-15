import xlwt
import os
import pandas as pd

def task_func(json_str, filename, sheet_name="sheet1"):
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError('json_str must be a string, bytes, or bytearray')
    
    try:
        data = pd.read_json(json_str)
    except ValueError:
        raise ValueError('Invalid JSON format')

    if data.empty:
        data = pd.DataFrame()

    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(sheet_name)

    if not data.empty:
        for r_idx, row in enumerate(data.to_numpy(), start=0):
            for c_idx, value in enumerate(row):
                worksheet.write(r_idx, c_idx, value)
        for c_idx, column in enumerate(data.columns, start=0):
            worksheet.write(0, c_idx, column)

    file_path = os.path.abspath(filename)
    workbook.save(file_path)
    return file_path