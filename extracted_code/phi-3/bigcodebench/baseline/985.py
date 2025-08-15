import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    if not json_data:
        raise ValueError("JSON data is empty")
    
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON data")
    
    if "Countries" not in data:
        raise ValueError("JSON data does not contain 'Countries' key")
    
    countries_data = data["Countries"]
    
    if not isinstance(countries_data, dict):
        raise ValueError("Country names are not strings or populations are not numeric")
    
    for country, population in countries_data.items():
        if not isinstance(country, str) or not (isinstance(population, int) or isinstance(population, float)):
            raise ValueError("Invalid country name or population value")
        if population < 0:
            raise ValueError("Negative population value")
    
    df = pd.DataFrame(list(countries_data.items()), columns=["Country", "Population"])
    df["Population"] = df["Population"].apply(lambda x: math.floor(x))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file_path = os.path.join(output_dir, file_name)
    
    try:
        df.to_csv(csv_file_path, index=False)
    except IOError as e:
        raise IOError(f"Failed to write CSV file: {e}")
    
    return csv_file_path, df