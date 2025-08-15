import pandas as pd
import json
import os
import math

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    # Load JSON data
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON data")

    # Validate and extract country data
    if "Countries" not in data:
        raise ValueError("JSON data must contain a 'Countries' key")
    
    countries_data = data["Countries"]
    
    if not isinstance(countries_data, dict):
        raise ValueError("'Countries' must be a dictionary")

    # Prepare data for DataFrame
    country_list = []
    population_list = []

    for country, population in countries_data.items():
        if not isinstance(country, str):
            raise ValueError("Country names must be strings")
        
        if not (isinstance(population, int) or isinstance(population, float)):
            raise ValueError("Population must be a number")
        
        if population < 0:
            raise ValueError("Population cannot be negative")
        
        # Round down float populations to the nearest integer
        population = math.floor(population)
        
        country_list.append(country)
        population_list.append(population)

    # Create DataFrame
    df = pd.DataFrame({
        "Country": country_list,
        "Population": population_list
    })

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define full file path
    file_path = os.path.join(output_dir, file_name)

    # Save DataFrame to CSV
    try:
        df.to_csv(file_path, index=False)
    except IOError:
        raise IOError("Could not write to the specified directory")

    return file_path, df