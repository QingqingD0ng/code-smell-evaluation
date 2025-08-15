import pandas as pd
import json
import os

def task_func(json_data, output_dir=".", file_name="country_population_report.csv"):
    try:
        data = json.loads(json_data)
        countries_data = data.get("Countries", {})
        
        if not isinstance(countries_data, dict):
            raise ValueError("Malformed JSON data: 'Countries' key must map to a dictionary.")
        
        countries = []
        populations = []
        
        for country, population in countries_data.items():
            if not isinstance(country, str):
                raise ValueError("Country names must be strings.")
            if not isinstance(population, (int, float)) or population < 0:
                raise ValueError("Populations must be non-negative numbers.")
            
            countries.append(country)
            populations.append(math.floor(population))
        
        df = pd.DataFrame({"Country": countries, "Population": populations})
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, file_name)
        df.to_csv(file_path, index=False)
        
        return file_path, df
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON data format.")
    except Exception as e:
        raise IOError(f"An error occurred while writing the file: {e}")