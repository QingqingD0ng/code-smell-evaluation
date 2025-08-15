import os

import re

from pathlib import Path


def task_func(dir_path: str, predicates: list) -> dict:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{dir_path} is not a valid directory")

    if not predicates:
        raise ValueError("No valid predicates provided")

    predicates = list(dict.fromkeys(predicates))  # Deduplicate predicates

    valid_conditions = {
        'is_file': lambda name: os.path.isfile(name),
        'is_dir': lambda name: os.path.isdir(name),
        'has_special_chars': lambda name: bool(re.search(r'[^a-zA-Z0-9_]', Path(name).stem)),
        'has_numbers': lambda name: bool(re.search(r'\d', name))
    }

    conditions = [valid_conditions[pred] for pred in predicates if pred in valid_conditions]

    results = {}

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        item_name = Path(item_path).name
        item_result = {cond: cond_func(item_name) for cond, cond_func in zip(conditions, conditions)}
        results[item_name] = item_result

    return results