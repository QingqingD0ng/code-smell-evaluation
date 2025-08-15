import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list) -> dict:
    valid_predicates = {
        'is_file': lambda p: p.is_file(),
        'is_dir': lambda p: p.is_dir(),
        'has_special_chars': lambda p: bool(re.search(r'\W', p.stem)),
        'has_numbers': lambda p: any(char.isdigit() for char in p.stem)
    }

    predicates = list(set(predicates))
    if not predicates or not all(p in valid_predicates for p in predicates):
        raise ValueError("No valid predicates provided.")

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The specified directory does not exist or is not a directory: {dir_path}")

    result = {}
    for item in os.listdir(dir_path):
        item_path = Path(dir_path) / item
        item_result = {pred: valid_predicates[pred](item_path) for pred in predicates}
        result[item] = item_result

    return result