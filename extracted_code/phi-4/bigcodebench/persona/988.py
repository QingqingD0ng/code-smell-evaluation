import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list) -> dict:
    # Define the supported predicates and their corresponding functions
    supported_predicates = {
        'is_file': lambda p: p.is_file(),
        'is_dir': lambda p: p.is_dir(),
        'has_special_chars': lambda name: bool(re.search(r'[^\w\s]', name)),
        'has_numbers': lambda name: bool(re.search(r'\d', name))
    }

    # Validate the directory
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist or is not a directory.")

    # Deduplicate and validate predicates
    valid_predicates = set(predicates) & supported_predicates.keys()
    if not valid_predicates:
        raise ValueError("No valid predicates provided.")

    # Prepare the results dictionary
    results = {}

    # Iterate over each item in the directory
    for item in os.listdir(dir_path):
        item_path = Path(dir_path) / item
        item_name = item_path.name

        # Evaluate each valid predicate
        results[item] = {predicate: func(item_name) for predicate, func in supported_predicates.items() if predicate in valid_predicates}

    return results