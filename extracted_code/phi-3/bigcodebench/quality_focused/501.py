import xlwt
import os
import pandas as pd
import json

def task_func(json_str, filename, sheet_name="sheet1"):
    # Check if json_str is a string, bytes, or bytearray
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError("json_str must be a string, bytes, or bytearray")

    # Try to load the JSON string
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")

    # Check if data is empty
    if not data:
        data = [{"Name": None, "Age": None}]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create the Excel file
    book = xlwt.Workbook()
    sheet = book.add_sheet(sheet_name)

    # Write data to Excel
    for i, row in df.iterrows():
        for j, value in enumerate(row):
            sheet.write(i, j, value)

    # Save the Excel file
    abs_path = os.path.abspath(filename)
    book.save(abs_path)

    return abs_path