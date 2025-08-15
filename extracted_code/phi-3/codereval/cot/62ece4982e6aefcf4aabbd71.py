import fnmatch
import re

def regex_dict(item):
    return {fnmatch.translate(k): v for k, v in item.items()}