import re

class Validator:
    def __init__(self):
        # Regular expression for identifier validation
        self.__pattern = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

    def is_valid(self, identifier):
        return bool(self.__pattern.match(identifier))