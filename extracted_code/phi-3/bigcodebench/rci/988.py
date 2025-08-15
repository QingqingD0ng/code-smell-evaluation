import os
import re
from pathlib import Path

def get_name_without_extension(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def check_conditions(item_path: str, conditions: list) -> dict:
    results = {}
    for condition in conditions:
        if condition == 'is_file':
            results[condition] = os.path.isfile(item_path)
        elif condition == 'is_dir':
            results[condition] = os.path.isdir(item_path)
        elif condition == 'has_special_chars':
            results[condition] = bool(re.search(r'\W', get_name_without_extension(item_path)))
        elif condition == 'has_numbers':
            results[condition] = bool(re.search(r'\d', get_name_without_extension(item_path)))
        else:
            raise ValueError(f"Invalid condition: {condition}")
    results['item_name'] = get_name_without_extension(item_path)
    return results

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
        results[item_path] = check_conditions(item_path,