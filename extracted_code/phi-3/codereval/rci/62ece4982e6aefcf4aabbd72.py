import re

def remove_quotes_from_name(name):
    # Check if the name starts with a double quote
    if name.startswith('"'):
        # Check if the name ends with a double quote
        if name.endswith('"'):
            # Remove the double quotes using regular expression
            return re.sub(r'^"(.*)"$', r'\1', name)
        else:
            # Return the original name if it doesn't end with a double quote
            return name
    # Check if the name starts with a single quote
    elif name.startswith("'"):
        # Check if the name ends with a single quote
        if name.endswith("'"):
            # Remove the single quotes using regular expression
            return re.sub(r"^'(.*)'$", r'\1', name)
        else:
            # Return the original name if it doesn't end with a single quote
            return name
    # Return the original name if it doesn't have quotes
    else:
        return name