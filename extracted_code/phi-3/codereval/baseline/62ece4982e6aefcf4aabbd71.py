import fnmatch
import re

def regex_dict(item):
    regex_dict = {}
    for key, value in item.items():
        regex_key = re.escape(key).replace(r'\*', '.*').replace(r'\?', '.')
        regex_dict[regex_key] = value
    return regex_dict