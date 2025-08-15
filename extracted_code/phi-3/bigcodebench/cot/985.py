import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    try:
        data = json.loads(json_data)
        if not isinstance(data, dict) or "Countries" not in data:
            raise ValueError("JSON data is malformed.")
        countries_data = data["Countries"]
        if not isinstance(countries_data, dict) or not all(isinstance(value, int) and value >= 0 for value in countries_data.values()):
            raise ValueError("Non-string country names or non-numeric or negative populations found.")
        df = pd.DataFrame(list(countries_data.items()), columns=["Country", "Population"])
        df["Population"] = df["Population"].apply(lambda x: math.floor(x))
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, file_name)
        df.to_csv(csv_file_path, index=False)
        return csv_file_path, df
    except (json.JSONDecodeError, ValueError, IOError) as e:
        raise e