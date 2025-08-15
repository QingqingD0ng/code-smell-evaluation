import re

def validate_value(value, pattern):
    return re.match(pattern, value)