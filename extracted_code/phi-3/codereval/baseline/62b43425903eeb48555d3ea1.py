import re

class YourClass:
    def _create_in_regex(self) -> re.Pattern:
        # Assuming the in-style parameter is a string starting with a letter or underscore, followed by any combination of letters, numbers, and underscores
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return re.compile(pattern)