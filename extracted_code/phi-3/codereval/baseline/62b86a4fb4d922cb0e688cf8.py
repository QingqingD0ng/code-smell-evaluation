import re

def validate_value(value, pattern):
    if re.match(pattern, value):
        return True
    else:
        return False