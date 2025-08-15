import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list) -> dict:
    valid_predicates = {'is_file', 'is_dir', 'has_special_chars', 'has_numbers'}
    predicates = set(predicates) & valid_predicates

    if not predicates:
        raise ValueError("No valid predicates provided.")

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The specified directory '{dir_path}' does not exist or is not a directory.")

    results = {}

    for item in os.listdir(dir_path):
        item_path = Path(dir_path) / item
        item_name = item.split('.')[0]  # Ignore file extension

        item_results = {}
        for predicate in predicates:
            if predicate == 'is_file':
                item_results[predicate] = item_path.is_file()
            elif predicate == 'is_dir':
                item_results[predicate] = item_path.is_dir()
            elif predicate == 'has_special_chars':
                item_results[predicate] = bool(re.search(r'\W', item_name))
            elif predicate == 'has_numbers':
                item_results[predicate] = bool(re.search(r'\d', item_name))

        results[item] = item_results

    return results