import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list):
    results = {}
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist or is not a directory.")
    
    predicates_dict = {
        'is_file': lambda name: os.path.isfile(os.path.join(dir_path, name)),
        'is_dir': lambda name: os.path.isdir(os.path.join(dir_path, name)),
        'has_special_chars': lambda name: bool(re.search(r'[^a-zA-Z0-9_]', name)),
        'has_numbers': lambda name: bool(re.search(r'\d', name))
    }
    
    valid_predicates = {predicate for predicate in predicates if predicate in predicates_dict}
    if not valid_predicates:
        raise ValueError("No valid predicates provided.")
    
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        item_results = {predicate: predicates_dict[predicate](item) for predicate in valid_predicates}
        results[item] = item_results
    
    return results