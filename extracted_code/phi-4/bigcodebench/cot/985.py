import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON data")

    if "Countries" not in data:
        raise ValueError("JSON data does not contain 'Countries' key")

    countries = data["Countries"]

    if not countries:
        raise ValueError("Empty country data")

    country_list = []
    population_list = []

    for country, population in countries.items():
        if not isinstance(country, str):
            raise ValueError("Country names must be strings")
        if not isinstance(population, (int, float)) or population < 0:
            raise ValueError("Population must be a non-negative number")
        
        country_list.append(country)
        population_list.append(math.floor(population))

    df = pd.DataFrame({"Country": country_list, "Population": population_list})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)

    try:
        df.to_csv(output_path, index=False)
    except IOError:
        raise IOError("Could not write to the specified directory")

    return output_path, df