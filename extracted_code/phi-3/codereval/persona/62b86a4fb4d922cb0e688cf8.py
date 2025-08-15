import re

def validate_value(value):
    # Define a dictionary of values and their corresponding regular expressions
    patterns = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^(http|https)://[a-zA-Z0-9./?=_-]+$',
        'phone': r'^\+?\d{10,15}$',
        'credit_card': r'^\d{4}-?\d{4}-?\d{4}-?\d{4}$'
    }
    
    # Check if the value matches any of the patterns
    for key, pattern in patterns.items():
        if re.match(pattern, value):
            return f"{value} is valid for {key}."
    return f"{value} is invalid or does not match any pattern."