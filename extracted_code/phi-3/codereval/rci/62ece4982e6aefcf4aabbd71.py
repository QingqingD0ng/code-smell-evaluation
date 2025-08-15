import fnmatch
import re

def regex_dict(item):
    if not isinstance(item, dict):
        raise ValueError("Input must be a dictionary.")

    regex_dict = {}
    for key in item:
        if not isinstance(key, str):
            raise ValueError("All keys must be strings.")
        regex_key = re.escape(key).replace(r'\*', '.*').replace(r'\?', '.')
        regex_dict[regex_key] = item[key]
    return regex_dict