import re

def _get_resource_name_regex():
    # Regular expression for valid Krake resource names
    # Resource names typically consist of lowercase letters, digits, and hyphens,
    # starting and ending with a letter or digit, and cannot start with a hyphen.
    resource_name_regex = r'^[a-z0-9][a-z0-9-]*[a-z0-9]$'
    return resource_name_regex

# Example usage:
regex = _get_resource_name_regex()
test_name = "valid-resource-name1"
if re.match(regex, test_name):
    print("Valid resource name")
else:
    print("Invalid resource name")