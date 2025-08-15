import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{dir_path} is not a directory or does not exist.")
    if not predicates:
        raise ValueError("No valid predicates provided.")

    conditions = {
        'is_file': lambda name: os.path.isfile(name),
        'is_dir': lambda name: os.path.isdir(name),
        'has_special_chars': lambda name: bool(re.search(r'\W', name)),
        'has_numbers': lambda name: bool(re.search(r'\d', name))
    }
    filtered_conditions = [cond for cond in predicates if cond in conditions]

    results = {}
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        item_name_without_ext = os.path.splitext(item)[0]
        result = {cond: conditions[cond](item_path) for cond in filtered_conditions}
        result.update({'item_name': item_name_without_ext})
        results[item_path] = result

    return results