import os
import re
from pathlib import Path


def task_func(dir_path: str, predicates: list) -> dict:
    if not predicates:
        raise ValueError("No valid predicates provided.")

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{dir_path} does not exist or is not a directory.")

    def is_special_char(s):
        return bool(re.search(r'[^a-zA-Z0-9_]', s))

    def has_numbers(s):
        return any(char.isdigit() for char in s)

    results = {}
    for item in os.listdir(dir_path):
        item_path = Path(dir_path) / item
        item_name = item_path.name

        results[item] = {
            'is_file': item_path.is_file(),
            'is_dir': item_path.is_dir(),
            'has_special_chars': is_special_char(item_name),
            'has_numbers': has_numbers(item_name)
        }

        for predicate in predicates:
            if predicate not in results[item]:
                results[item][predicate] = False
            else:
                results[item][predicate] = results[item][predicate]

    return results