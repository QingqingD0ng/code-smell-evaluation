import re

class ParameterPatternGenerator:
    def _create_in_regex(self) -> re.Pattern:
        return re.compile(r'\b\w+\b')