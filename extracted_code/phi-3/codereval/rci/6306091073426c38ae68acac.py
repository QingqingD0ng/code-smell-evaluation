from functools import reduce
from typing import Any, Dict

def set_nested_value(dic: Dict[Any, Any], keys: list, val: Any) -> None:
    """
    Set a value for a nested key in a dictionary.
    """
    if not keys or not isinstance(dic, dict):
        return

    current_dict = dic
    for key in keys[:-1]:
        current_dict = current_dict.setdefault(key, {})

    current_dict[keys[-1]] = val

def dict_insert(dic: Dict[Any, Any], val: Any, key: str, *keys: str) -> None:
    """
    Insert a value of a nested key into a dictionary.
    """
    key_list = key.split('.')
    set_nested_value(dic, key_list + keys, val)