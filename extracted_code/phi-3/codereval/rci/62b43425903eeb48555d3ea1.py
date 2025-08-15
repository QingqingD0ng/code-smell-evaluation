class YourClass:
    _compiled_in_regex = None

    def _create_in_regex(self) -> re.Pattern:
        if not self._compiled_in_regex:
            pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
            try:
                self._compiled_in_regex = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return self._compiled_in_regex