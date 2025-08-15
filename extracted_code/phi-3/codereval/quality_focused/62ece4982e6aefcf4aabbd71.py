import re

def regex_dict(item):
    regex_dict = {}
    for key, value in item.items():
        regex_dict[re.escape(key).replace(r'\*', '.*').replace(r'\?', '.')] = value
    return regex_dict