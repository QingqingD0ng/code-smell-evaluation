import json

def task_func(json_str, filename, sheet_name="sheet1"):
    try:
        # Convert JSON string to Python object
        data = json.loads(json_str)
        if not data:  # Handle empty JSON arrays
            data = []

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create an Excel writer object
        excel_path = os.path.abspath(filename)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

        # Write DataFrame to an Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Save the Excel file
        writer.save()

        return excel_path
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON provided.")
    except TypeError:
        raise TypeError("Invalid type for JSON string. Expected string, bytes, or bytearray.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")