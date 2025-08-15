import xlwt
import os
import pandas as pd
import json

def task_func(json_str, filename, sheet_name="sheet1"):
    if not isinstance(json_str, (str, bytes, bytearray)):
        raise TypeError("json_str must be a string, bytes, or bytearray.")
    
    try:
        # Decode if json_str is bytes or bytearray
        if isinstance(json_str, (bytes, bytearray)):
            json_str = json_str.decode('utf-8')
        
        # Load JSON data
        data = json.loads(json_str)
        
        # Convert JSON data to DataFrame
        if data:
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
        
        # Create an Excel workbook and sheet
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet(sheet_name)
        
        # Write DataFrame to Excel sheet
        for r, row in enumerate(df.to_numpy()):
            for c, value in enumerate(row):
                sheet.write(r, c, value)
        
        # Save the workbook
        workbook.save(filename)
        
        # Return the absolute path of the created Excel file
        return os.path.abspath(filename)
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Example usage:
# json_str = '[{"Name": "John", "Age": 30}, {"Name": "Jane", "Age": 28}]'
# print(task_func(json_str, 'data.xls').endswith('data.xls'))  # True
# os.remove('data.xls')