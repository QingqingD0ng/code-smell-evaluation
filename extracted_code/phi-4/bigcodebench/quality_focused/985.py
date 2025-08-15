import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON data")

    if not isinstance(data, dict) or "Countries" not in data:
        raise ValueError("JSON data must contain a 'Countries' key")

    countries_data = data["Countries"]
    if not isinstance(countries_data, dict):
        raise ValueError("'Countries' must be a dictionary")

    records = []
    for country, population in countries_data.items():
        if not isinstance(country, str):
            raise ValueError("Country names must be strings")

        if not isinstance(population, (int, float)):
            raise ValueError("Population must be a number")

        if math.isnan(population) or population < 0:
            raise ValueError("Population must be a non-negative number")

        population = math.floor(population) if isinstance(population, float) else population
        records.append({"Country": country, "Population": population})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid country data found")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    try:
        df.to_csv(output_path, index=False)
    except IOError:
        raise IOError("Could not write to the specified directory")

    return output_path, df