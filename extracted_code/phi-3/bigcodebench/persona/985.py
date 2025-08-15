import pandas as pd

import json

import os


def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):

    # Convert JSON to dict

    data = json.loads(json_data)

    if "Countries" not in data:

        raise ValueError("JSON data does not contain 'Countries' key.")


    # Validate and prepare data

    countries = []

    for country, population in data["Countries"].items():

        if not isinstance(country, str):

            raise ValueError("Country names must be strings.")

        if not isinstance(population, (int, float)) or population < 0:

            raise ValueError("Populations must be non-negative integers or floats.")

        countries.append((country, math.floor(population)))


    # Create DataFrame

    df = pd.DataFrame(countries, columns=["Country", "Population"])


    # Create output directory if it doesn't exist

    os.makedirs(output_dir, exist_ok=True)


    # Save DataFrame to CSV

    csv_file_path = os.path.join(output_dir, file_name)

    df.to_csv(csv_file_path, index=False)


    return csv_file_path, df