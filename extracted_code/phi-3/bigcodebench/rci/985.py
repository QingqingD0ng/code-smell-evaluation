import pandas as pd
import json
import os
import math
from typing import Tuple, Dict

def parse_json(json_data: str) -> Dict:
    try:
        return json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON data")

def validate_countries_data(data: Dict):
    if "Countries" not in data:
        raise ValueError("JSON data does not contain 'Countries' key")
    if not isinstance(data["Countries"], dict):
        raise ValueError("Country names are not strings or population values are not numeric")
    for country, population in data["Countries"].items():
        if not isinstance(country, str) or not (isinstance(population, int) or isinstance(population, float)):
            raise ValueError("Invalid country name or population value")
        if population < 0:
            raise ValueError("Negative population value")

def create_directory_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_file(dataframe: pd.DataFrame, output_dir: str, file_name: str):
    csv_file_path = os.path.join(output_dir, file_name)
    try:
        dataframe.to_csv(csv_file_path, index=False)
    except IOError as e:
        raise IOError(f"Failed to write CSV file: {e}")

def task_func(json_data: str, output_dir: str = ".", file_name: str = "country_population_report.csv") -> Tuple[str, pd.DataFrame]:
    if not json_data:
        raise ValueError("JSON data is empty")
    
    data = parse_json(json_data)
    validate_countries_data(data)

    df = pd.DataFrame(list(data["Countries"].items()), columns=["Country", "Population"])
    df