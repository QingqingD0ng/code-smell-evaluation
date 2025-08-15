import os
import re
from pathlib import Path

def task_func(dir_path: str, predicates: list) -> dict:
    valid_predicates = {
        'is_file': lambda p: p.is_file(),
        'is_dir': lambda p: p.is_dir(),
        'has_special_chars': lambda p: bool(re.search(r'[^a-zA-Z0-9_]', p.stem)),
        'has_numbers': lambda p: bool(re.search(r'\d', p.stem))
    }
    
    if not Path(dir_path).is_dir():
        raise FileNotFoundError(f"The path '{dir_path}' does not exist or is not a directory.")
    
    predicates = list(set(predicates))  # Deduplicate predicates
    predicates = [pred for pred in predicates if pred in valid_predicates]
    
    if not predicates:
        raise ValueError("No valid predicates provided.")
    
    results = {}
    for item in os.listdir(dir_path):
        item_path = Path(dir_path) / item
        result = {pred: valid_predicates[pred](item_path) for pred in predicates}
        results[item] = result
    
    return results