import re

class RegexCreator:
    def __init__(self):
        pass

    def _create_in_regex(self) -> re.Pattern:
        pattern_to_match = r'\bparam\d+\b'
        return re.compile(pattern_to_match)

# Example usage:
# regex_creator = RegexCreator()
# in_regex_pattern = regex_creator._create_in_regex()
# matches = in_regex_pattern.findall(some_text)