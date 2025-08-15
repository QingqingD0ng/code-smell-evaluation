import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    if not json_data:
        raise ValueError("JSON data is empty")

    try:
        json_dict = json.loads(json_data)
        if "Countries" not in json_dict or not isinstance(json_dict["Countries"], dict):
            raise ValueError("JSON data is malformed")

        data = []
        for country, population in json_dict["Countries"].items():
            if not isinstance(country, str) or not isinstance(population, (int, float)):
                raise ValueError("Country names must be strings and populations must be numeric")
            if population < 0:
                raise ValueError("Populations cannot be negative")
            data.append({"Country": country, "Population": math.floor(population)})

        df = pd.DataFrame(data)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_file_path = os.path.join(output_dir, file_name)
        df.to_csv(csv_file_path, index=False)

        return csv_file_path, df
    except Exception as e:
        raise IOError(f"Error writing CSV file: {e}")

# Example usage:
# json_str = '{"Countries": {"Country A": 331002651, "Country B": 67886011}}'
# csv_file_path, df = task_func(json_str)
# print(csv_file_path)
# print(df)